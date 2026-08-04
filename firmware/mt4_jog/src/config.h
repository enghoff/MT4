#ifndef MT4_CONFIG_H
#define MT4_CONFIG_H

#include <Arduino.h>

#include "kinematics.h"

// Timer1/3 run at 16 MHz / prescaler 8 = 2 MHz, i.e. 2 ticks per us.
static const uint16_t TIMER_TICKS_PER_US = 2;

struct StepPinIO {
  volatile uint8_t *port;
  uint8_t mask;
};

enum PinModeSetting : uint8_t { PIN_FLOAT = 0, PIN_LOW = 1, PIN_HIGH = 2 };

static const uint8_t LAB_PINS[] = {
    22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37,
    40,
};
static const uint8_t LAB_PIN_COUNT = sizeof(LAB_PINS) / sizeof(LAB_PINS[0]);
static const uint8_t ENABLE_PIN = 40;

static const uint8_t LIMIT_PINS[] = {20, 21};
static const uint8_t LIMIT_PIN_COUNT = 2;

static const uint16_t STEP_PULSE_US = 10;
static const uint16_t JOG_STEP_PERIOD_MIN_US = 700;
static const uint16_t JOG_STEP_PERIOD_MAX_US = 4000;
static const uint16_t CJ_REFRESH_MS = 40;
/* `mp` absolute moves: linear TCP interpolation step size (mm) and cap on
 * the number of segments per move (longer moves use a larger effective step). */
static const float MP_CART_SEGMENT_MM = 2.0f;
static const uint16_t MP_MAX_SEGMENTS = 250;
/* Sampling spacing/cap for route_joints_feasible()'s upfront IK + soft-limit
 * preflight check on a candidate route -- deliberately coarser than
 * MP_CART_SEGMENT_MM (which governs the actually-executed segment
 * granularity). This check is a fail-fast convenience, not the only safety
 * net: every segment's target is independently re-checked against soft
 * limits at execution time (motion.cpp's mp_execute_segment), and every
 * individual step is guarded again in the Timer1 ISR
 * (motion_step_allowed) -- so a coarser preflight can't let anything
 * through that isn't still caught (safely aborting the move) at execution
 * time. Soft-limit envelope violations are broad, slowly-varying regions
 * (not needle-thin), so coarser sampling still catches them in practice.
 * Measured live: at MP_CART_SEGMENT_MM's old 2mm/40-sample-cap density,
 * every `mp` call cost 65-110ms of blocking IK time regardless of the
 * move's actual complexity -- comparable to a tracker's own ~100ms tick
 * period, causing commands to queue up faster than firmware could drain
 * them. */
static const float MP_FEASIBILITY_SAMPLE_MM = 8.0f;
static const uint8_t MP_FEASIBILITY_MAX_SAMPLES = 20;
/* Keep-out cylinder around the J1 axis (any Z): the TCP physically cannot
 * get closer to the base column than roughly this. `mp` targets inside it
 * are rejected; `mp` paths that would cross it are routed around it
 * (tangent-arc-tangent, shortest side); Cartesian jog clamps the inward
 * velocity component at the boundary so jogging slides along the cylinder
 * instead of driving into the base. */
static const float MT4_KEEPOUT_RADIUS_MM = 140.0f;
/* Desk / ground plane: TCP Z below this is rejected (`mp`) and Cartesian jog
 * clamps downward velocity. Desk contact is ~120mm in this home-angle frame
 * (107/−9.3), so the floor sits ~5mm under it for headroom. Anyone jogging
 * down to feel for the desk hits this clamp first, which is indistinguishable
 * from contact -- measure the surface with a camera-tracked descent instead. */
static const float MT4_GROUND_Z_MM = 115.0f;
/* Ground / keep-out guard poll for joint-space motion (`j` jog, `m` moves) --
 * see refresh_envelope_guard_if_due() in motion.cpp for why this is a poll and
 * not a per-step ISR check. 10ms is 4x tighter than the Cartesian jog refresh
 * because this guard is the ONLY thing standing between a joint jog and the
 * desk, where cj also has its own setup-time clamp. At the fastest jog rate
 * (700us/step) 10ms is ~14 steps, about 1.3mm of TCP travel at a typical
 * 200mm reach, so the stop lands roughly a millimetre before the plane rather
 * than after it. */
static const uint16_t MT4_ENVELOPE_GUARD_MS = 10;
/* Lookahead bounds in steps. The floor keeps a very slow jog from looking at
 * a pose indistinguishable from the current one (which would never trip);
 * the ceiling keeps a fast one from refusing motion far from the envelope. */
static const int32_t MT4_ENVELOPE_LOOKAHEAD_MIN = 6;
static const int32_t MT4_ENVELOPE_LOOKAHEAD_MAX = 200;
/* Soft joint step limits. J2/J3 counters are relative to the
 * limit/interference reference (steps=0 at switch / J3 fold-into-J2).
 * The J2/J3 numbers are the envelope maxima measured 2026-07-19 plus
 * (1000, 500), the offset between the park-zero reference they were taken in
 * and this limit reference (same physical poses). J1 switch-side min is
 * overwritten at do_home()
 * from the center distance (limit = -j1_center); J2 switch-side min is
 * forced to 0. */
static const int32_t MT4_JOINT_SOFT_MIN_DEFAULT[MT4_NUM_JOINTS] = {
    -4800L, 0L, -1550L, -8100L};
static const int32_t MT4_JOINT_SOFT_MAX_DEFAULT[MT4_NUM_JOINTS] = {
    4580L, 3950L, 1650L, 8100L};
/* Coupled J2+J3 extension limit (step counters, limit-referenced).
 * Because J2 and J3 have opposite step signs, j2_deg - j3_deg =
 * const - (j2_steps + j3_steps)/spd, so a *minimum* link-angle difference
 * at full stretch is a *maximum* on j2_steps + j3_steps. The park-zero
 * in-sample max sum is 2910; +1500 for the (1000+500) reference shift
 * → 4410. Soft min on the sum is loose — the folded extreme is gated by
 * J3 min + ground Z. */
static const int32_t MT4_J2_J3_SUM_MAX = 4410L;
static const int32_t MT4_J2_J3_SUM_MIN = -200L;
/* Max path pieces for a routed `mp` move: radial onto a route cylinder +
 * entry tangent + arc + exit tangent (+ optional radial in). */
static const uint8_t MP_MAX_PIECES = 5;
/* `mq` waypoint queue: pending legs behind whatever is currently executing
 * (an `mp`/`mq` leg already in flight). Sized well above a typical
 * pick/stack cycle (roughly 4-6 legs); each entry is a handful of bytes, so
 * this is cheap on SRAM even at this headroom. */
static const uint8_t MQ_QUEUE_CAPACITY = 8;
/* When the keep-out arc at R=140 would violate soft joint limits (common
 * at high Z), `mp` retries the same tangent-arc-tangent topology at these
 * larger radii until a joint-feasible route is found. */
static const float MT4_ROUTE_RADIUS_MM[] = {140.0f, 165.0f, 190.0f, 215.0f,
                                           240.0f, 265.0f};
static const uint8_t MT4_ROUTE_RADIUS_COUNT =
    sizeof(MT4_ROUTE_RADIUS_MM) / sizeof(MT4_ROUTE_RADIUS_MM[0]);
/* `mp` acceleration ramp (dda.cpp): a move starts at this slower, safe-to-
 * start-from-rest step period and ramps toward the move's cruise speed over
 * up to this many master ticks, then symmetrically ramps back up to
 * MP_ACCEL_START_US before the move ends. No-ops (plain constant-speed
 * stepping) when the requested cruise speed is already this
 * slow or slower, or the move is too short for a full ramp -- see
 * dda_set_ramp(). Untuned against real stall/skip behavior yet; the values
 * below are a conservative starting point for reaching the 700us max. */
static const uint16_t MP_ACCEL_START_US = 1800;
static const uint16_t MP_ACCEL_RAMP_TICKS = 60;
/* Generous headroom; ~2 full sweeps of any joint at current steps/deg. */
static const long MOVE_MAX_STEPS = 100000L;
static const uint16_t HOME_STEP_PERIOD_US = 800;
static const uint16_t ENABLE_SETTLE_MS = 5;
static const bool ENABLE_ACTIVE_LOW = true;
static const bool LIMIT_ACTIVE_LOW = true;

static const uint8_t J1_DRIVE = 23;
static const uint8_t J1_DIR = 22;
static const uint8_t J1_LIMIT = 21;
static const uint8_t J2_DRIVE = 25;
static const uint8_t J2_DIR = 24;
static const uint8_t J2_LIMIT = 20;
static const uint8_t J3_DRIVE = 27;
static const uint8_t J3_DIR = 26;
static const uint8_t J4_DRIVE = 35;
static const uint8_t J4_DIR = 36;
static const uint8_t J_DRIVE[MT4_NUM_JOINTS] = {J1_DRIVE, J2_DRIVE, J3_DRIVE,
                                                J4_DRIVE};
static const uint8_t J_DIR_PIN[MT4_NUM_JOINTS] = {J1_DIR, J2_DIR, J3_DIR,
                                                  J4_DIR};
static const bool J_DIR_POS_HIGH[MT4_NUM_JOINTS] = {false, false, false,
                                                    false};
static const bool J1_HOME_DIR_HIGH = true;
static const bool J2_HOME_DIR_HIGH = true;
// J3 has no limit switch of its own. This is the direction that drives it
// into mechanical interference with J2 (which, at J2's raw limit trigger,
// displaces J2 enough to release J2's OWN limit switch) -- used as J3's
// end-of-travel reference. The opposite direction pulls both joints off.
static const bool J3_HOME_DIR_HIGH = true;
static const uint16_t J1_HOME_CENTER_DEFAULT = 4580;
static const uint16_t J2_HOME_PULL_DEFAULT = 1000;
static const uint16_t J3_HOME_PULL_DEFAULT = 500;
// Pre-home backoff for J3 so it isn't already sitting in a position that
// interferes with J2's approach to its own limit switch.
static const uint16_t J3_PREWIDEN_STEPS = 1000;
// J2 widen steps after releasing its limit switch (if already
// triggered after J3 pre-widen) before seeking back toward the switch.
static const uint16_t J2_PREWIDEN_STEPS = 500;
static const uint32_t HOME_SEEK_MAX = 25000;
// Dwell after J1 hits its limit switch, before reversing back toward
// center. Lets J1 fully settle at the switch (no ramp/creep on approach)
// instead of reversing direction immediately.
static const uint16_t J1_HOME_LIMIT_PAUSE_MS = 300;

#endif // MT4_CONFIG_H
