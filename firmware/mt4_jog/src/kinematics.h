#ifndef MT4_KINEMATICS_H
#define MT4_KINEMATICS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MT4_NUM_JOINTS 4
#define MT4_CJ_MASTER 10000L

/* +1 if positive step count increases firmware joint angle, -1 if inverted.
 * J3 confirmed inverted 2026-07-06: a +299-step probe raised the forearm
 * tip rather than lowering it, which is what -1.0 predicts (photo-confirmed). */
#define MT4_J1_STEP_SIGN 1.0f
#define MT4_J2_STEP_SIGN (-1.0f)
#define MT4_J3_STEP_SIGN 1.0f
#define MT4_J4_STEP_SIGN 1.0f

/* Model angles at step-counter zero. J1/J4 zero at the post-home park
 * (J1 centered; J4 jaws-along-arm after j4zero). J2/J3 zero at the
 * limit/interference reference so pull-off changes do not invalidate the
 * angle fit: after do_home(), counters sit at +(j2_pull, j3_pull).
 *
 * J2/J3 values derived from the 2026-07-21 tape-fit park pose (107.0°,
 * −9.3°) under the pull-offs then in effect (1000 / 500 steps):
 *   ref = park - SIGN * pull / SPD
 * FK at park still reports TCP ≈ (190.0, 0, 225.6). The 1000/500 literals
 * below are frozen fit provenance — do not retie to J*_HOME_PULL_DEFAULT. */
#define MT4_HOME_J1_DEG 0.0f
#define MT4_HOME_J2_DEG (107.0f + 1000.0f / 35.0f) /* ≈135.571° at J2 switch */
#define MT4_HOME_J3_DEG (-9.3f - 500.0f / 35.0f)   /* ≈−23.586° at J3 interfer */
#define MT4_HOME_J4_DEG 0.0f

extern const float MT4_STEPS_PER_DEG[MT4_NUM_JOINTS];

typedef struct {
  float j1;
  float j2;
  float j3;
  float j4;
} JointAnglesDeg;

typedef struct {
  float x;
  float y;
  float z;
} Vec3;

typedef struct {
  int32_t j1;
  int32_t j2;
  int32_t j3;
  int32_t j4;
  int32_t master;
} CartesianRates;

void mt4_fk_tcp(const JointAnglesDeg *q, Vec3 *out);

/* Gripper yaw in world frame (deg). Invariant when J4 counters J1 1:1
 * (`orient on` / mp world-space hold): ws_j4 = j4_joint + j1. */
float mt4_ws_j4_deg(const JointAnglesDeg *q);

bool mt4_cartesian_rates(const JointAnglesDeg *q, const Vec3 *dir_unit,
                         bool hold_orient, CartesianRates *out);

/* Closed-form position IK: TCP (x, y, z) mm, origin at the base under J1's
 * pivot -> joint angles. J4 is taken directly (absolute), not solved. Elbow
 * branch and J1 wrap are chosen nearest `near`. Returns false if (x, y, z)
 * is outside the two-link reach. Mirrors mt4_jog/kinematics.py's
 * ik_position(). */
bool mt4_ik_position(const JointAnglesDeg *near, float x, float y, float z,
                     float j4_deg, JointAnglesDeg *out);

#ifdef __cplusplus
}
#endif

#endif
