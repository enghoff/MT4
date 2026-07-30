#include "commands.h"

#include <Arduino.h>
#include <math.h>

#include "config.h"
#include "gripper.h"
#include "homing.h"
#include "kinematics.h"
#include "motion.h"
#include "pins.h"

static bool parse_pin(const char *token, uint8_t *out) {
  if (!token || !*token) {
    return false;
  }
  if (token[0] == 'd' || token[0] == 'D') {
    ++token;
  }
  const long pin = atol(token);
  if (pin < 2 || pin > 53) {
    return false;
  }
  *out = static_cast<uint8_t>(pin);
  return true;
}

static bool parse_cj_vec(const char *arg, Vec3 *out, int8_t *j4_roll) {
  *j4_roll = 0;
  while (*arg == ' ') {
    ++arg;
  }
  if (*arg == '\0') {
    return false;
  }

  // Single-axis shorthand: an optional sign followed by x/y/z (e.g. "-x").
  // Peek at a separate cursor so a leading '-' on the general signed-triple
  // form below (e.g. "-1 0 0") is NOT consumed here.
  int sign = 1;
  const char *p = arg;
  if (*p == '+' || *p == '-') {
    sign = (*p == '-') ? -1 : 1;
    ++p;
  }

  if ((p[0] == 'x' || p[0] == 'X') && p[1] == '\0') {
    out->x = static_cast<float>(sign);
    out->y = 0.0f;
    out->z = 0.0f;
    return true;
  }
  if ((p[0] == 'y' || p[0] == 'Y') && p[1] == '\0') {
    out->x = 0.0f;
    out->y = static_cast<float>(sign);
    out->z = 0.0f;
    return true;
  }
  if ((p[0] == 'z' || p[0] == 'Z') && p[1] == '\0') {
    out->x = 0.0f;
    out->y = 0.0f;
    out->z = static_cast<float>(sign);
    return true;
  }

  // General signed triple with an optional 4th J4-roll direction, parsed
  // from the ORIGINAL (unstripped) arg so each component keeps its own
  // sign. The roll rides along with the Cartesian jog (rotate the wrist
  // while translating); a zero direction with nonzero roll is a pure
  // J4 roll through the same code path.
  long dx = 0;
  long dy = 0;
  long dz = 0;
  long roll = 0;
  const int n = sscanf(arg, "%ld %ld %ld %ld", &dx, &dy, &dz, &roll);
  if (n >= 3) {
    out->x = static_cast<float>(dx);
    out->y = static_cast<float>(dy);
    out->z = static_cast<float>(dz);
    if (n == 4) {
      *j4_roll = (roll > 0) ? 1 : (roll < 0 ? -1 : 0);
    }
    return fabsf(out->x) + fabsf(out->y) + fabsf(out->z) > 1e-6f ||
           *j4_roll != 0;
  }
  return false;
}

// Parses "<x> <y> <z> <j4|h|w> <g> [speed_us] [dwell_ms]" (mp/mq command
// args) without relying on sscanf's %f -- avr-libc's default sscanf doesn't
// support float conversion unless linked against a non-default scanf flavor,
// which this build isn't. strtod/strtol are always available. The j4 slot
// also accepts the sentinels `h` (hold world yaw) / `w` (hold wrist joint) --
// see Mt4J4Mode in motion.h; *j4 is left 0 for those. speed_us defaults to 0
// (leave the current jog/move step period unchanged); otherwise clamped to
// [700, 4000] us. dwell_ms defaults to 0 and only means anything to `mq`,
// where >0 makes the entry a grip station (see motion_queue_move).
static bool parse_mp_args(char *arg, float *x, float *y, float *z, float *j4,
                          uint8_t *j4_mode, long *g, long *speed_us,
                          long *dwell_ms) {
  float vals[4];
  uint8_t mode = MT4_J4_EXPLICIT;
  for (uint8_t i = 0; i < 4; ++i) {
    while (*arg == ' ') {
      ++arg;
    }
    if (!*arg) {
      return false;
    }
    if (i == 3 && (arg[1] == ' ' || arg[1] == '\0')) {
      const char c = arg[0];
      if (c == 'h' || c == 'H' || c == 'w' || c == 'W') {
        mode = (c == 'h' || c == 'H') ? MT4_J4_HOLD : MT4_J4_WRIST;
        vals[3] = 0.0f;
        ++arg;
        continue;
      }
    }
    char *end = nullptr;
    vals[i] = strtod(arg, &end);
    if (end == arg) {
      return false;
    }
    arg = end;
  }
  while (*arg == ' ') {
    ++arg;
  }
  if (!*arg) {
    return false;
  }
  char *end = nullptr;
  const long gv = strtol(arg, &end, 10);
  if (end == arg) {
    return false;
  }
  arg = end;
  *x = vals[0];
  *y = vals[1];
  *z = vals[2];
  *j4 = vals[3];
  *j4_mode = mode;
  *g = gv;
  *speed_us = 0;
  *dwell_ms = 0;
  while (*arg == ' ') {
    ++arg;
  }
  if (!*arg) {
    return true;
  }
  end = nullptr;
  const long sv = strtol(arg, &end, 10);
  if (end == arg) {
    return false;
  }
  *speed_us = sv;
  arg = end;
  while (*arg == ' ') {
    ++arg;
  }
  if (!*arg) {
    return true;
  }
  end = nullptr;
  const long dv = strtol(arg, &end, 10);
  if (end == arg) {
    return false;
  }
  *dwell_ms = dv;
  arg = end;
  while (*arg == ' ') {
    ++arg;
  }
  if (*arg) {
    return false;
  }
  return true;
}

void handle_line(char *line) {
  while (*line == ' ' || *line == '\t') {
    ++line;
  }
  if (*line == '\0') {
    return;
  }

  /* Any command outside this exemption list safely stops an active jog
   * first (e.g. "?", "e0") -- but gripper commands ("g o"/"g c"/"g stop"/
   * "g <S>") don't touch TCP motion and used to be caught by this net too,
   * silently killing cartesian jog every time the gripper was actuated
   * while jogging. Exempt them so the gripper sweeps concurrently with an
   * active `cj` jog instead of interrupting it. Same story for "pos": it's
   * a read-only FK query polled every tick by cj-based trackers specifically
   * as a non-blocking, non-disruptive status read (unlike "?", which also
   * dumps mode/limits/gripper and isn't polled in a hot loop) -- without
   * this exemption it silently killed the jog on the very next tick after
   * every re-engage, which read as the tracker barely moving at all.
   * "mp " is exempted for the same reason "cj " is: a tracker that
   * re-targets an absolute move every tick (see start_absolute_move's
   * in-flight retarget path) needs those `mp` lines to reach
   * start_absolute_move() with jog_active still true so it can detect the
   * retarget and splice in the new path instead of hard-stopping first --
   * stop_jog() here would zero jog_active before start_absolute_move() ever
   * gets a look, defeating the smoothing unconditionally. "mq " needs the
   * same exemption: motion_queue_move() decides "queue this behind what's
   * running" vs. "nothing's running, cold-start it now" from jog_active --
   * stop_jog() here would zero that first and make every `mq` look idle,
   * so nothing would ever actually get queued. "?"/"d" are pure status
   * reads and must be passive like "pos" already is: a host polling
   * progress (or resolving state) mid-path must not halt the path -- that
   * un-exempted "?" is exactly what used to force clients to choose
   * between knowing the arm's state and keeping it moving. */
  if (strcmp(line, "!") && strcmp(line, "stop") && strcmp(line, "j") &&
      strcmp(line, "jog") && strcmp(line, "home") && strcmp(line, "$H") &&
      strncmp(line, "home ", 5) && strncmp(line, "cj ", 3) &&
      strncmp(line, "mp ", 3) && strncmp(line, "mq ", 3) &&
      strncmp(line, "speed ", 6) && strcmp(line, "pos") &&
      strcmp(line, "?") && strcmp(line, "d") &&
      line[0] != 'g' && line[0] != 'G') {
    stop_jog();
  }

  if (!strcmp(line, "?") || !strcmp(line, "d")) {
    print_status();
    return;
  }
  if (!strcmp(line, "s")) {
    print_limits();
    return;
  }
  if (line[0] == 'g' || line[0] == 'G') {
    const char *arg = line + 1;
    while (*arg == ' ') {
      ++arg;
    }
    if (*arg == '\0') {
      printGripperStatus();
      return;
    }
    if (!strcmp(arg, "stop") || !strcmp(arg, "0")) {
      gripperSweepStop();
      Serial.println(F("ok grip stop"));
      return;
    }
    if (!strcmp(arg, "o") || !strcmp(arg, "open")) {
      gripperSweepStart(GRIP_SWEEP_OPEN);
      if (gripper_s <= GRIPPER_S_OPEN && gripper_sweep == GRIP_SWEEP_STOP) {
        Serial.println(F("ok grip at open"));
      } else {
        Serial.println(F("ok grip open"));
      }
      return;
    }
    if (!strcmp(arg, "c") || !strcmp(arg, "close")) {
      gripperSweepStart(GRIP_SWEEP_CLOSE);
      if (gripper_s >= GRIPPER_S_CLOSED && gripper_sweep == GRIP_SWEEP_STOP) {
        Serial.println(F("ok grip at closed"));
      } else {
        Serial.println(F("ok grip close"));
      }
      return;
    }
    long v = atol(arg);
    if (!gripperSValid(v)) {
      Serial.print(F("err grip stop|o|c|"));
      Serial.print(GRIPPER_S_OPEN);
      Serial.print('-');
      Serial.println(GRIPPER_S_CLOSED);
      return;
    }
    setGripperS(static_cast<uint16_t>(v));
    Serial.println(F("ok grip"));
    return;
  }
  if (!strcmp(line, "home") || !strcmp(line, "$H")) {
    do_home(J1_HOME_CENTER_DEFAULT, J2_HOME_PULL_DEFAULT);
    return;
  }
  if (!strncmp(line, "home ", 5)) {
    long j1 = atol(line + 5);
    long j2 = J2_HOME_PULL_DEFAULT;
    char *rest = strchr(line + 5, ' ');
    if (rest) {
      while (*rest == ' ') {
        ++rest;
      }
      j2 = atol(rest);
    }
    if (j1 <= 0 || j1 > 20000 || j2 <= 0 || j2 > 20000) {
      Serial.println(F("err home <j1> <j2>"));
      return;
    }
    do_home(static_cast<uint16_t>(j1), static_cast<uint16_t>(j2));
    return;
  }
  if (!strcmp(line, "pos")) {
    print_joint_pos();
    return;
  }
  if (!strncmp(line, "setpos ", 7)) {
    long v[MT4_NUM_JOINTS] = {0, 0, 0, 0};
    if (sscanf(line + 7, "%ld %ld %ld %ld", &v[0], &v[1], &v[2], &v[3]) != 4) {
      Serial.println(F("err setpos <j1> <j2> <j3> <j4>"));
      return;
    }
    motion_set_joint_steps(v);
    return;
  }
  if (!strcmp(line, "j4zero")) {
    motion_zero_j4_world();
    return;
  }
  if (!strncmp(line, "speed ", 6)) {
    const char *arg = line + 6;
    while (*arg == ' ') {
      ++arg;
    }
    char *end = nullptr;
    long us = strtol(arg, &end, 10);
    if (end == arg || *end != '\0') {
      Serial.println(F("err speed <us>"));
      return;
    }
    motion_set_speed_us(us);
    return;
  }
  if (!strncmp(line, "orient ", 7)) {
    const char *arg = line + 7;
    while (*arg == ' ') {
      ++arg;
    }
    if (!strcmp(arg, "on") || !strcmp(arg, "hold")) {
      cart_orient_hold = true;
    } else if (!strcmp(arg, "off") || !strcmp(arg, "free")) {
      cart_orient_hold = false;
    } else {
      Serial.println(F("err orient on|off"));
      return;
    }
    Serial.print(F("ok orient "));
    Serial.println(cart_orient_hold ? F("hold") : F("free"));
    return;
  }
  if (!strncmp(line, "cj ", 3)) {
    Vec3 dir = {0.0f, 0.0f, 0.0f};
    int8_t j4_roll = 0;
    if (!parse_cj_vec(line + 3, &dir, &j4_roll)) {
      Serial.println(F("err cj +x|-x|+y|-y|+z|-z|dx dy dz [j4]"));
      return;
    }
    start_cartesian_jog(dir, j4_roll);
    return;
  }
  if ((line[0] == 'm' || line[0] == 'M') && (line[1] == 'p' || line[1] == 'P') &&
      line[2] == ' ') {
    float x, y, z, j4;
    uint8_t j4_mode;
    long g, speed_us, dwell_ms;
    if (!parse_mp_args(line + 3, &x, &y, &z, &j4, &j4_mode, &g, &speed_us,
                       &dwell_ms)) {
      Serial.println(F("err mp <x> <y> <z> <j4|h|w> <g> [speed_us]"));
      return;
    }
    if (dwell_ms != 0) {
      /* A grip station only means something inside a queue -- a lone `mp`
       * has nothing to hold back. Reject rather than silently ignore. */
      Serial.println(F("err mp dwell (use mq)"));
      return;
    }
    start_absolute_move(x, y, z, j4, j4_mode, g, speed_us);
    return;
  }
  if ((line[0] == 'm' || line[0] == 'M') && (line[1] == 'q' || line[1] == 'Q') &&
      line[2] == ' ') {
    float x, y, z, j4;
    uint8_t j4_mode;
    long g, speed_us, dwell_ms;
    if (!parse_mp_args(line + 3, &x, &y, &z, &j4, &j4_mode, &g, &speed_us,
                       &dwell_ms)) {
      Serial.println(
          F("err mq <x> <y> <z> <j4|h|w> <g> [speed_us] [dwell_ms]"));
      return;
    }
    motion_queue_move(x, y, z, j4, j4_mode, g, speed_us, dwell_ms);
    return;
  }
  if ((line[0] == 'm' || line[0] == 'M') && line[1] == ' ') {
    long d[MT4_NUM_JOINTS] = {0, 0, 0, 0};
    long dg = 0;
    const int n = sscanf(line + 2, "%ld %ld %ld %ld %ld",
                         &d[0], &d[1], &d[2], &d[3], &dg);
    if (n < 4) {
      Serial.println(F("err m <dj1> <dj2> <dj3> <dj4> [dg]"));
      return;
    }
    if (n == 4) {
      /* avr-libc's sscanf does not reliably leave a trailing unmatched %ld
       * argument untouched -- force the optional gripper delta to 0 rather
       * than trust whatever it wrote. */
      dg = 0;
    }
    for (uint8_t i = 0; i < MT4_NUM_JOINTS; ++i) {
      if (d[i] > MOVE_MAX_STEPS || d[i] < -MOVE_MAX_STEPS) {
        Serial.println(F("err m step delta too large"));
        return;
      }
    }
    /* GRIPPER_S_* are uint16_t: their difference promotes to unsigned int,
     * and negating an unsigned value wraps instead of going negative (165
     * became 65371, not -165) -- cast to a signed type first. */
    const long gripper_span =
        static_cast<long>(GRIPPER_S_CLOSED) - static_cast<long>(GRIPPER_S_OPEN);
    if (dg > gripper_span || dg < -gripper_span) {
      Serial.println(F("err m gripper delta too large"));
      return;
    }
    start_relative_move(d, dg);
    return;
  }
  if (!strcmp(line, "j") || !strcmp(line, "jog")) {
    motion_start_jog();
    return;
  }
  if (!strcmp(line, "!") || !strcmp(line, "stop")) {
    stop_jog();
    motion_cancel_move();
    Serial.println(F("ok stop"));
    return;
  }
  if (!strcmp(line, "all f")) {
    stop_jog();
    motion_cancel_move();  // drop any in-flight/queued mp/mq path too
    apply_all(PIN_FLOAT);
    drivers_enabled = false;
    clear_jog_axes();
    Serial.println(F("ok all float"));
    return;
  }
  if (!strcmp(line, "e0")) {
    set_enable(false);
    Serial.println(F("ok enable off"));
    return;
  }
  if (!strcmp(line, "e1")) {
    set_enable(true);
    Serial.println(F("ok enable on"));
    return;
  }
  if (line[0] == 'x' || line[0] == 'X') {
    const bool add = line[1] == '+';
    const bool remove = line[1] == '-';
    const char *tok = line + 1;
    if (add || remove) {
      ++tok;
    }
    if (!strcmp(tok, "c") || !strcmp(tok, "C")) {
      clear_jog_axes();
      Serial.println(F("ok step clear"));
      return;
    }
    uint8_t pin = 0;
    if (!parse_pin(tok, &pin)) {
      Serial.println(F("err x<pin>|x+<pin>|x-<pin>|xc"));
      return;
    }
    if (remove) {
      if (!remove_jog_pin(pin)) {
        Serial.println(F("err step missing"));
        return;
      }
    } else if (add) {
      if (!add_jog_pin(pin)) {
        Serial.println(F("err step full"));
        return;
      }
    } else {
      clear_jog_axes();
      if (!add_jog_pin(pin)) {
        Serial.println(F("err step"));
        return;
      }
    }
    Serial.println(F("ok step"));
    return;
  }
  if ((line[0] == 'd' || line[0] == 'D') && line[1] >= '0' && line[1] <= '9') {
    char *sp = strchr(line, ' ');
    if (!sp) {
      Serial.println(F("err d<pin> f|l|h"));
      return;
    }
    *sp = '\0';
    uint8_t pin = 0;
    if (!parse_pin(line, &pin)) {
      Serial.println(F("err d<pin> f|l|h"));
      return;
    }
    const char *m = sp + 1;
    while (*m == ' ') {
      ++m;
    }
    PinModeSetting mode;
    if (!strcmp(m, "f")) {
      mode = PIN_FLOAT;
    } else if (!strcmp(m, "l")) {
      mode = PIN_LOW;
    } else if (!strcmp(m, "h")) {
      mode = PIN_HIGH;
    } else {
      Serial.println(F("err mode f|l|h"));
      return;
    }
    const int8_t idx = lab_index(pin);
    if (idx >= 0) {
      apply_pin(static_cast<uint8_t>(idx), mode);
    } else {
      pinMode(pin, mode == PIN_FLOAT ? INPUT : OUTPUT);
      if (mode != PIN_FLOAT) {
        digitalWrite(pin, mode == PIN_HIGH ? HIGH : LOW);
      }
    }
    if (pin == step_pin && mode == PIN_FLOAT) {
      remove_jog_pin(pin);
    }
    Serial.println(F("ok pin"));
    return;
  }

  Serial.println(F("err unknown"));
}
