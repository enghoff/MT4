#ifndef MT4_MOTION_H
#define MT4_MOTION_H

#include <Arduino.h>

#include "config.h"
#include "kinematics.h"

// Cartesian wrist-unwind on/off (`orient on|off` command); plain data, no
// hardware-register concerns like the DDA/speed state in dda.h. When on,
// J4 counters base yaw 1:1 (dq4 = -dq1).
extern bool cart_orient_hold;

// True once do_home() has completed successfully this session; never reset
// except by a power cycle/reflash. Gates the `mp` absolute-position command.
extern bool mt4_homed;

// Soft joint step limits and ground plane. Defaults from config.h; J1
// switch-side min (-center) and J2 min (0 = limit) are refreshed at do_home().
extern int32_t joint_soft_min[MT4_NUM_JOINTS];
extern int32_t joint_soft_max[MT4_NUM_JOINTS];
extern float mt4_ground_z_mm;

void motion_init();
void reset_joint_steps();
/* setpos command: directly overwrite the joint step counters (no motion). */
void motion_set_joint_steps(const long steps[MT4_NUM_JOINTS]);
/* j4zero command: rewrite J4's step counter so the *current* physical wrist
 * pose reports world-frame J4 = 0 (joint_j4 = -j1). No motion. After this,
 * jaws aligned with the arm at j1=0 read as world j4=0; face-align picks can
 * treat offset=0. Survives subsequent `home` (J4 is not re-zeroed there). */
void motion_zero_j4_world();

/* Install defaults, then (after home) set J1 switch-side min from
 * j1_center and J2 min to 0 (limit-referenced). j2/j3_pull are unused for
 * limits (pull only sets post-home park counters). Prints `home limits ...`. */
void motion_apply_home_soft_limits(uint16_t j1_center, uint16_t j2_pull,
                                   uint16_t j3_pull);
/* True when every joint step counter is inside [soft_min, soft_max]. */
bool motion_joints_within_soft_limits(const long steps[MT4_NUM_JOINTS]);
/* True when a single step in `positive` direction on `joint` stays inside. */
bool motion_step_allowed(uint8_t joint, bool positive);

void stop_jog();
/* "!"/"stop" command: stop_jog() plus canceling any in-progress `m` move
 * bookkeeping (so a stopped move doesn't later report "m done"). */
void motion_cancel_move();
void print_status();
void print_joint_pos();

/* Legacy single/multi-axis jog (x/x+/x-/xc/j commands). */
void clear_jog_axes();
bool add_jog_pin(uint8_t pin);
bool remove_jog_pin(uint8_t pin);
/* "j"/"jog" command: starts the jog ISR if any legacy axis is armed. */
bool motion_start_jog();

/* Cartesian jog (`cj` command): resolved-rate world-frame TCP jog, on-device
 * Jacobian/DLS solve re-run every CJ_REFRESH_MS while active. j4_roll
 * (-1/0/+1) adds a wrist roll on top of the solved rates (including on top
 * of the orient-hold counter-rotation), so J4 can rotate while the TCP
 * translates; a zero dir with nonzero roll is a pure J4 roll. */
void start_cartesian_jog(Vec3 dir, int8_t j4_roll);
void refresh_cartesian_jog_if_due();
/* Ground/keep-out guard for `j` jog and `m` moves; call every loop(). */
void refresh_envelope_guard_if_due();

/* Bounded relative move (`m` command): all joints move proportionally so
 * they finish together; completion is reported asynchronously from loop(). */
void start_relative_move(const long d[MT4_NUM_JOINTS], long dg);
/* Call every loop() iteration; prints "m done pos ..." and stops the jog
 * exactly once when a relative move completes. */
void motion_poll_move_done();

/* How `mp`/`mq` resolve the J4 field. EXPLICIT: the wire value is a
 * world-frame yaw in degrees. HOLD (`h` on the wire): resolve to the
 * world yaw the arm has when the leg is PLANNED (cold start / retarget:
 * now; queued leg: when it's popped and the arm is wherever the previous
 * leg left it) -- continuous world-orientation hold with no host probe.
 * WRIST (`w`): keep the J4 *joint* angle fixed across the leg's J1 swing
 * (world yaw rotates with the base) -- the firmware-native version of the
 * host-side j4_preserve_wrist(), which big base swings need because a
 * world-yaw hold there commands joint J4 = world - j1 past its soft
 * limits. Both sentinels resolve per leg at plan time, which is what
 * makes them usable for queued waypoints (the pose a queued leg starts
 * from isn't known at enqueue time). */
enum Mt4J4Mode : uint8_t {
  MT4_J4_EXPLICIT = 0,
  MT4_J4_HOLD = 1,
  MT4_J4_WRIST = 2,
};

/* "mp" command: move to an absolute TCP position (x, y, z mm, origin at the
 * base under J1's pivot) + J4 field (world-frame gripper yaw in deg, or the
 * sentinels `h`/`w` -- see Mt4J4Mode; j4_deg is ignored unless j4_mode is
 * MT4_J4_EXPLICIT) + absolute gripper S (0 = leave the gripper alone) +
 * optional speed_us (700-4000, same units as `speed`; 0 = leave the current
 * step period unchanged). Rejected with "err not homed" unless mt4_homed.
 * TCP xyz is interpolated along straight world-frame lines in short
 * segments. When the resolved J4 matches the current world-frame yaw,
 * gripper orientation is held fixed in world space (J4 counters J1 1:1);
 * otherwise world-frame J4 is interpolated linearly.
 * Async "mp done pos ..." matches `m`. */
bool start_absolute_move(float x, float y, float z, float j4_deg,
                         uint8_t j4_mode, long g, long speed_us);

/* "mq" command: enqueue a Cartesian waypoint (same args/validation as `mp`,
 * including the `h`/`w` J4 sentinels -- resolved when the leg is planned,
 * not when it's enqueued) behind whatever is currently executing. If the arm is idle, the waypoint
 * starts immediately -- a cold start identical to `mp`, just acknowledged
 * "ok mq" instead of "ok mp". Otherwise it's appended to a small pending
 * queue (MQ_QUEUE_CAPACITY deep; "err mq full" beyond that) and picked up,
 * without stopping, when the leg currently running finishes its segments --
 * same no-stop splice `mp`'s in-flight retarget already uses, but (unlike a
 * live retarget) each queued leg's route is *fully* replanned with the same
 * keep-out/soft-limit feasibility sweep a cold-start `mp` runs, since queued
 * waypoints are host-planned jumps (e.g. around a cube stack), not a live
 * tracker's small incremental corrections -- skipping that sweep would let
 * the firmware substitute its own unchecked straight chord instead of the
 * route the host actually vetted. Every queued leg (and the currently
 * executing one, if it came off the queue) still individually ramps down
 * near its own end before ramping back up for the next -- this removes the
 * per-waypoint stop/settle/full-reaccel cycle and serial round trip, but
 * does not (yet) give one flat-cruise ramp across the whole queue. An `mp`
 * sent while a queue is in flight keeps its existing override behavior
 * (immediate retarget to the new target) and drops any still-pending queued
 * legs. Async completion of the *whole* queue reuses "mp done pos ...".
 *
 * dwell_ms > 0 makes the entry a GRIP STATION instead of a leg: no motion is
 * planned, the gripper is driven to `g`, and the queue holds until the sweep
 * finishes plus dwell_ms of mechanical settle, then resumes on its own. This
 * is what lets a whole pick-and-place be one queue -- the per-leg `g` fires
 * at leg *start* and sweeps while the arm moves, which is wrong for a grasp
 * (the jaws must close after the descent, with the arm stopped). A station's
 * x/y/z must equal the pose the leg ahead of it ended at; a mismatch beyond a
 * few mm is "err mq station pose" and kills the path, rather than gripping
 * somewhere the host did not intend. Capped at MQ_STATION_DWELL_MAX_MS
 * ("err mq dwell 0-N") since a station stalls the whole queue.
 * `mp` rejects a nonzero dwell outright -- a station is only meaningful
 * inside a queue. */
bool motion_queue_move(float x, float y, float z, float j4_deg,
                       uint8_t j4_mode, long g, long speed_us,
                       long dwell_ms);

/* Polled from the main loop: releases a pending grip-station hold once the
 * gripper sweep has finished and the settle has elapsed, then resumes the
 * queue. No-op when no station is pending. */
void motion_poll_mq_hold();

/* "speed <us>" command: clamps, applies to the timer live, and prints the
 * accepted value. */
void motion_set_speed_us(long us);

#endif // MT4_MOTION_H
