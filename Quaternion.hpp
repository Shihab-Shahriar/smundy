#ifndef Quaternion_H
#define Quaternion_H

#include <Kokkos_Core.hpp>
#include "smath.hpp"

struct Quaternion {
  // w,x,y,z
  double w;
  double x;
  double y;
  double z;

  // constructors
  Quaternion(const Vec<double, 4> &q)
  {
    w = q[0];
    x = q[1];
    y = q[2];
    z = q[3];
  }

  Quaternion(const double qw, const double qx, const double qy, const double qz)
  {
    w = qw;
    x = qx;
    y = qy;
    z = qz;
  }

  Quaternion(const Vec<double, 3> &v, const double sina_2, const double cosa_2)
  {
    from_rot(v, sina_2, cosa_2);
  }

  Quaternion(const Vec<double, 3> &v, const double angle) { from_rot(v, angle); }

  KOKKOS_INLINE_FUNCTION Quaternion(const double u1, const double u2, const double u3) { from_unit_random(u1, u2, u3); }

  KOKKOS_INLINE_FUNCTION Quaternion() { from_unit_random(); }

  // quaternion from rotation around a given axis (given sine and cosine of HALF the rotation angle)
  void from_rot(const Vec<double, 3> &v, const double sina_2, const double cosa_2)
  {
    w = cosa_2;
    x = sina_2 * v[0];
    y = sina_2 * v[1];
    z = sina_2 * v[2];
  }

  // rotation around a given axis (angle without range restriction)
  void from_rot(const Vec<double, 3> &v, const double angle)
  {
    const double sina_2 = Kokkos::sin(angle / 2);
    const double cosa_2 = Kokkos::cos(angle / 2);
    w = cosa_2;
    x = sina_2 * v[0];
    y = sina_2 * v[1];
    z = sina_2 * v[2];
  }

  // set a unit random quaternion representing uniform distribution on sphere surface
  KOKKOS_INLINE_FUNCTION void from_unit_random(const double u1, const double u2, const double u3)
  {
    // a random unit quaternion following a uniform distribution law on SO(3)
    // from three U[0,1] random numbers
    constexpr double pi = 3.14159265358979323846;
    const double a = Kokkos::sqrt(1 - u1);
    const double b = Kokkos::sqrt(u1);
    const double su2 = Kokkos::sin(2 * pi * u2);
    const double cu2 = Kokkos::cos(2 * pi * u2);
    const double su3 = Kokkos::sin(2 * pi * u3);
    const double cu3 = Kokkos::cos(2 * pi * u3);
    w = a * su2;
    x = a * cu2;
    y = b * su3;
    z = b * cu3;
  }

  // set a unit random quaternion representing uniform distribution on sphere surface
  KOKKOS_INLINE_FUNCTION void from_unit_random()
  {
    // non threadsafe random unit quaternion
    const double u1 = 1.0; //(double) rand() / RAND_MAX;
    const double u2 = 2.0; //(double) rand() / RAND_MAX;
    const double u3 = 0.0; //(double) rand() / RAND_MAX;
    from_unit_random(u1, u2, u3);
  }

  // normalize the quaternion q / ||q||
  void normalize()
  {
    const double norm = Kokkos::sqrt(w * w + x * x + y * y + z * z);
    w = w / norm;
    x = x / norm;
    y = y / norm;
    z = z / norm;
  }

  // rotate a point v in 3D space around the origin using this quaternion
  // see EN Wikipedia on Quaternions and spatial rotation
  Vec<double, 3> rotate(const Vec<double, 3> &v) const
  {
    const double t2 = x * y;
    const double t3 = x * z;
    const double t4 = x * w;
    const double t5 = -y * y;
    const double t6 = y * z;
    const double t7 = y * w;
    const double t8 = -z * z;
    const double t9 = z * w;
    const double t10 = -w * w;
    return Vec<double, 3>({double(2.0) * ((t8 + t10) * v[0] + (t6 - t4) * v[1] + (t3 + t7) * v[2]) + v[0],
        double(2.0) * ((t4 + t6) * v[0] + (t5 + t10) * v[1] + (t9 - t2) * v[2]) + v[1],
        double(2.0) * ((t7 - t3) * v[0] + (t2 + t9) * v[1] + (t5 + t8) * v[2]) + v[2]});
  }

  // rotate a point v in 3D space around a given point p using this quaternion
  Vec<double, 3> rotate_around_point(const Vec<double, 3> &v, const Vec<double, 3> &p)
  {
    return rotate(v - p) + p;
  }

  /**
   * @brief rotate the quaternion itself based on rotational velocity omega
   *
   * Delong, JCP, 2015, Appendix A eq1, not linearized
   * @param q
   * @param omega rotational velocity
   * @param dt time interval
   */
  void rotate_self(const Vec<double, 3> &rot_vel, const double dt)
  {
    const double rot_vel_norm = Kokkos::sqrt(rot_vel[0] * rot_vel[0] + rot_vel[1] * rot_vel[1] + rot_vel[2] * rot_vel[2]);
    if (rot_vel_norm < std::numeric_limits<double>::epsilon()) {
      return;
    }
    const double rot_vel_norm_inv = 1.0 / rot_vel_norm;
    const double sw = sin(rot_vel_norm * dt / 2);
    const double cw = cos(rot_vel_norm * dt / 2);
    const double rot_vel_cross_xyz_0 = rot_vel[1] * z - rot_vel[2] * y;
    const double rot_vel_cross_xyz_1 = rot_vel[2] * x - rot_vel[0] * z;
    const double rot_vel_cross_xyz_2 = rot_vel[0] * y - rot_vel[1] * x;
    const double rot_vel_dot_xyz = rot_vel[0] * x + rot_vel[1] * y + rot_vel[2] * z;

    x = w * sw * rot_vel[0] * rot_vel_norm_inv + cw * x + sw * rot_vel_norm_inv * rot_vel_cross_xyz_0;
    y = w * sw * rot_vel[1] * rot_vel_norm_inv + cw * y + sw * rot_vel_norm_inv * rot_vel_cross_xyz_1;
    z = w * sw * rot_vel[2] * rot_vel_norm_inv + cw * z + sw * rot_vel_norm_inv * rot_vel_cross_xyz_2;
    w = w * cw - rot_vel_dot_xyz * sw * rot_vel_norm_inv;
    normalize();
  }

  /**
   * @brief rotate the quaternion itself based on rotational velocity omega
   *
   * Delong, JCP, 2015, Appendix A eq1, not linearized
   * @param q
   * @param omega rotational velocity
   * @param dt time interval
   */
  void rotate_self(const double rot_vel_x, const double rot_vel_y, const double rot_vel_z, const double dt)
  {
    const double rot_vel_norm = Kokkos::sqrt(rot_vel_x * rot_vel_x + rot_vel_y * rot_vel_y + rot_vel_z * rot_vel_z);
    if (rot_vel_norm < std::numeric_limits<double>::epsilon()) {
      return;
    }
    const double rot_vel_norm_inv = 1.0 / rot_vel_norm;
    const double sw = sin(rot_vel_norm * dt / 2);
    const double cw = cos(rot_vel_norm * dt / 2);
    const double rot_vel_cross_xyz_0 = rot_vel_y * z - rot_vel_z * y;
    const double rot_vel_cross_xyz_1 = rot_vel_z * x - rot_vel_x * z;
    const double rot_vel_cross_xyz_2 = rot_vel_x * y - rot_vel_y * x;
    const double rot_vel_dot_xyz = rot_vel_x * x + rot_vel_y * y + rot_vel_z * z;

    x = w * sw * rot_vel_x * rot_vel_norm_inv + cw * x + sw * rot_vel_norm_inv * rot_vel_cross_xyz_0;
    y = w * sw * rot_vel_y * rot_vel_norm_inv + cw * y + sw * rot_vel_norm_inv * rot_vel_cross_xyz_1;
    z = w * sw * rot_vel_z * rot_vel_norm_inv + cw * z + sw * rot_vel_norm_inv * rot_vel_cross_xyz_2;
    w = w * cw - rot_vel_dot_xyz * sw * rot_vel_norm_inv;
    normalize();
  }
};  // Quaternion


#endif
