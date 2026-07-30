/*
 * MT4 jog firmware — 4-axis step/dir jog + J1/J2 limit homing + Cartesian jog.
 * Serial @ 115200 (host: DTR/RTS off).
 *
 * Joint jog (legacy):
 *   all f | e0 | e1
 *   d<pin> f|l|h    direction / float
 *   x<pin> | x+<pin> | x-<pin> | xc   step pin(s) for jog
 *   j | stop          start/stop Timer1 jog ISR (equal step rate on active axes)
 *   speed <us>        set the shared jog step period (joint jog + cj + mp),
 *                        live-adjustable mid-jog; clamped to [700,4000] us
 *                        (session state, reset on reboot)
 *
 * Cartesian jog:
 *   cj +x|-x|+y|-y|+z|-z   world-frame TCP jog (multi-axis DDA on device)
 *   cj <dx> <dy> <dz> [j4] direction vector (integer components, normalized)
 *                            + optional J4 roll -1|0|1 layered on top of the
 *                            solved rates (incl. orient hold), so the wrist
 *                            rotates while the TCP moves; all-zero dir with
 *                            nonzero j4 = pure wrist roll. At the keep-out
 *                            cylinder (see below) the inward velocity
 *                            component is clamped so the jog slides along
 *                            the boundary instead of hitting the base.
 *
 * Keep-out: the TCP cannot physically approach the base column closer than
 * ~MT4_KEEPOUT_RADIUS_MM (140mm) from the J1 axis at any Z. `mp` rejects
 * targets inside the cylinder ("err mp keepout") and automatically routes
 * paths that would cross it (or whose keep-out graze would violate soft
 * joint limits) via tangent-arc-tangent on the smallest feasible cylinder
 * radius; a start inside the cylinder first escapes radially.
 * Soft joint limits (envelope; J2/J3 limit-referenced) and ground plane
 * MT4_GROUND_Z_MM also reject/clamp out-of-range jog and `mp`.
 * Joint-space moves (`m`, homing) are NOT covered -- they command raw steps.
 *   orient on|off          J4 wrist unwind when J1 moves (default on, 1:1)
 *   pos                      print joint step counters (since last home),
 *                              plus a derived "tcp x=.. y=.. z=.. j4=..
 *                              grip=.. speed=.." line (mm/deg/S/us -- x/y/z,
 *                              j4, and speed use the same units `mp` accepts)
 *   setpos <j1> <j2> <j3> <j4>
 *       Directly overwrite the joint step counters (no motion) -- for
 *       correcting drift after an external reference (e.g. a soft-contact
 *       seek on an unreferenced joint like J3, which has no limit switch).
 *   j4zero
 *       Rewrite J4's step counter so the current physical wrist pose reports
 *       world-frame J4 = 0 (no motion). Used after the operator aligns the
 *       jaws with the arm axis; subsequent face-align picks assume offset 0.
 *       Survives `home` (J4 counter is preserved across homing). Lost on
 *       power cycle / reflash until re-run.
 *
 * Relative move (bounded, coordinated):
 *   m <dj1> <dj2> <dj3> <dj4> [dg]
 *       Move each joint by the signed number of steps relative to the current
 *       position (multi-axis DDA, proportional rates, all joints finish
 *       together) and sweep the gripper by dg S-units relative to its current
 *       S (clamped to S120-285). Replies "ok m" on accept, then an async
 *       "m done pos ..." line when the joint motion completes. Drivers are
 *       left ENABLED (holding) after the move.
 *
 * Absolute move (bounded, coordinated, Cartesian-linear):
 *   mp <x> <y> <z> <j4|h|w> <g> [speed_us]
 *       Move to an absolute TCP position in mm (origin at the base, under
 *       J1's pivot) with a J4 field and an absolute gripper S (0 = leave
 *       the gripper alone). The J4 field is a world-frame gripper yaw in
 *       degrees, or one of two sentinels resolved at leg-plan time (see
 *       Mt4J4Mode in motion.h): `h` holds the world yaw the arm has when
 *       the leg is planned (continuous orientation hold, no host probe
 *       needed); `w` holds the J4 *joint* angle across the leg's J1 swing
 *       (world yaw follows the base -- what big swings need, since a
 *       world hold there drives joint J4 = world - j1 past soft limits).
 *       Optional speed_us sets the shared step period for the move
 *       (700-4000 us, same units as `speed`; 0 or omitted = leave the
 *       current period unchanged). TCP xyz is interpolated along straight
 *       world-frame lines in short segments; each segment solves
 *       closed-form IK (nearest branch to the current pose) and runs a
 *       coordinated joint DDA move. When the resolved J4 matches the pose
 *       at move start, gripper yaw is held fixed in world space (J4
 *       counters J1 1:1, like `orient on`); otherwise J4 is interpolated
 *       linearly to the target. Rejected with "err not homed" unless
 *       `home`/`$H` has completed successfully this session -- absolute
 *       coordinates are meaningless against an unreferenced step counter.
 *       Same "ok mp" / async "mp done pos ..." reply convention as `m`.
 *
 * Queued absolute move (multi-waypoint path, no per-waypoint round trip):
 *   mq <x> <y> <z> <j4|h|w> <g> [speed_us] [dwell_ms]
 *       Same arguments/validation as `mp` (sentinel J4 modes included --
 *       for a queued leg they resolve when the leg is POPPED and planned,
 *       against wherever the previous leg actually ended, which is what
 *       makes per-leg wrist behavior possible without host round trips). If the arm is idle this behaves
 *       exactly like `mp` (cold start), just acknowledged "ok mq". If an
 *       `mp`/`mq` move is already executing, the waypoint is appended to a
 *       small pending queue instead (MQ_QUEUE_CAPACITY deep; "err mq full
 *       N" beyond that, "ok mq queued N" on accept) and picked up -- without
 *       stopping -- the moment the leg currently running finishes its
 *       segments, the same no-stop splice `mp`'s in-flight retarget uses.
 *       Unlike a live `mp` retarget, each queued leg gets the *full*
 *       keep-out/soft-limit route feasibility check a cold-start `mp` runs
 *       (queued waypoints are host-planned jumps, e.g. routing around a
 *       cube stack, not a live tracker's small corrections); a leg that
 *       fails that check aborts the whole remaining queue, not just itself.
 *       Every leg still ramps down near its own end before ramping back up
 *       for the next (no flat single cruise across the whole queue yet) --
 *       what this removes is the per-waypoint stop/settle/reaccel cycle and
 *       serial round trip, executing exactly the route the host queued
 *       instead of the firmware re-deriving its own chord mid-flight. `mp`
 *       sent while a queue is in flight keeps its existing override
 *       behavior (retarget to the new target immediately) and drops
 *       whatever was still queued. The whole queue's completion reuses
 *       "mp done pos ...", same as a plain `mp`.
 *
 *       dwell_ms > 0 makes the entry a GRIP STATION rather than a leg: no
 *       motion is planned, the gripper is driven to <g>, and the queue holds
 *       until the sweep state machine reports done PLUS dwell_ms of
 *       mechanical settle, then resumes itself. This is what lets an entire
 *       pick-and-place be one queue. The per-leg <g> cannot do it: that
 *       fires at leg *start* and sweeps while the arm is moving, whereas a
 *       grasp needs the jaws to close after the descent, with the arm
 *       stopped. A station's x/y/z must equal the pose the leg ahead of it
 *       ended at (checked, "err mq station pose want ... at ..."), so a host
 *       that queued against a pose the arm never reached fails loudly
 *       instead of gripping in the wrong place; dwell is capped
 *       ("err mq dwell 0-N") because a station holds the whole queue. A
 *       station with <g> 0 is a pure timed pause. `?` shows HOLD=sweep or
 *       HOLD=settle while one is pending. `mp` rejects a nonzero dwell
 *       ("err mp dwell (use mq)") -- a lone move has nothing to hold back.
 *
 *   home [j1 j2]    widen J2/J3 off their min-angle extremes, home J1 (seek
 *                     I21, return to center), seek J2 to its raw I20
 *                     trigger, drive J3 into interference with J2 until I20
 *                     releases (J3's indirect end-of-travel reference, since
 *                     it has no limit switch of its own), then pull J2/J3
 *                     off (arg/default j2; J3 uses its own shorter default),
 *                     set J2/J3 counters to +pull (limit-referenced zeros),
 *                     and rotate J4 to its calibrated zero (step counter → 0;
 *                     after `j4zero` that is jaws-along-arm / world J4=0 at
 *                     the homed pose). J4's counter is preserved across the
 *                     J1–J3 rewrite so this move is meaningful.
 *   g o | g c | g stop   gripper sweep open/close (S120–S285 on device)
 *   g <120-285>           set S clamped to limits (manual)
 *   ? | s           status / limits
 *
 * Source layout: config.h (pin map/timing constants), pins.{h,cpp} (raw I/O,
 * limit switches), gripper.{h,cpp} (PWM gripper), motion.{h,cpp} (DDA/ISR
 * jog engine, cartesian + relative-move logic), homing.{h,cpp} (do_home),
 * commands.{h,cpp} (serial line parser), kinematics.{h,cpp} (FK/IK, on-
 * device Cartesian rate solve).
 */

#include <Arduino.h>

#include "commands.h"
#include "gripper.h"
#include "homing.h"
#include "motion.h"
#include "pins.h"

void setup() {
  pins_init();
  motion_init();
  gripperPwmInit();

  Serial.begin(115200);
  delay(400);
  reset_joint_steps();
  Serial.println(F("MT4 jog firmware ready (joint + cartesian)"));
  print_status();
}

void loop() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      line_buf[line_len] = '\0';
      handle_line(line_buf);
      line_len = 0;
      continue;
    }
    if (line_len < sizeof(line_buf) - 1) {
      line_buf[line_len++] = c;
    }
  }
  poll_limits();
  gripperSweepTick();
  refresh_cartesian_jog_if_due();
  motion_poll_move_done();
  /* After gripperSweepTick() and motion_poll_move_done(), so a station armed
   * this same pass sees the sweep's own state, and so the leg ahead of it has
   * already been retired. */
  motion_poll_mq_hold();
}
