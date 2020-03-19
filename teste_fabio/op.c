#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

struct dataobj
{
  void *restrict data;
  int * size;
  int * npsize;
  int * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
} ;

struct profiler
{
  double section0;
  double section1;
  double section2;
  double section3;
  double section4;
} ;


int Kernel(struct dataobj *restrict damp_vec, const float dt, const float o_x, const float o_y, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict usave_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int abc_y_left_ltkn, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler * timers)
{
  float (*restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]]) damp_vec->data;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  float (*restrict usave)[usave_vec->size[1]][usave_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[usave_vec->size[1]][usave_vec->size[2]]) usave_vec->data;
  float (*restrict vp)[vp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]]) vp_vec->data;
  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  for (int time = time_m, t0 = (time)%(3), t1 = (time + 1)%(3), t2 = (time + 2)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 1)%(3), t2 = (time + 2)%(3))
  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(damp,u,vp:32)
      for (int y = y_m; y <= y_M; y += 1)
      {
        float r3 = dt*dt;
        float r2 = vp[x + 2][y + 2]*vp[x + 2][y + 2];
        float r0 = r2*damp[x + 1][y + 1];
        float r1 = r2;
        u[t1][x + 2][y + 2] = 3.19999986e-3F*(1.5625e+2F*r0*dt*u[t2][x + 2][y + 2] - 5.0e+1F*r1*r3*u[t0][x + 2][y + 2] + 1.25e+1F*(r1*r3*u[t0][x + 1][y + 2] + r1*r3*u[t0][x + 2][y + 1] + r1*r3*u[t0][x + 2][y + 3] + r1*r3*u[t0][x + 3][y + 2]) + 6.25e+2F*u[t0][x + 2][y + 2] - 3.125e+2F*u[t2][x + 2][y + 2])/(5.0e-1F*r0*dt + 1.0F);
      }
    }
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    /* Begin section1 */
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
    {
      float r5 = (int)(floor(-2.0e-1F*o_y + 2.0e-1F*src_coords[p_src][1]));
      float r4 = (int)(floor(-2.0e-1F*o_x + 2.0e-1F*src_coords[p_src][0]));
      int ii_src_1 = r5;
      int ii_src_2 = r5 + 1;
      float py = (float)(-5.0F*r5 - o_y + src_coords[p_src][1]);
      int ii_src_0 = r4;
      int ii_src_3 = r4 + 1;
      float px = (float)(-5.0F*r4 - o_x + src_coords[p_src][0]);
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1)
      {
        float r6 = 7.05599955940247e-1F*(vp[ii_src_0 + 2][ii_src_1 + 2]*vp[ii_src_0 + 2][ii_src_1 + 2])*(4.0e-2F*px*py - 2.0e-1F*px - 2.0e-1F*py + 1)*src[time][p_src];
        u[t1][ii_src_0 + 2][ii_src_1 + 2] += r6;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_2 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= y_M + 1)
      {
        float r7 = 7.05599955940247e-1F*(vp[ii_src_0 + 2][ii_src_2 + 2]*vp[ii_src_0 + 2][ii_src_2 + 2])*(-4.0e-2F*px*py + 2.0e-1F*py)*src[time][p_src];
        u[t1][ii_src_0 + 2][ii_src_2 + 2] += r7;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_3 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= x_M + 1)
      {
        float r8 = 7.05599955940247e-1F*(vp[ii_src_3 + 2][ii_src_1 + 2]*vp[ii_src_3 + 2][ii_src_1 + 2])*(-4.0e-2F*px*py + 2.0e-1F*px)*src[time][p_src];
        u[t1][ii_src_3 + 2][ii_src_1 + 2] += r8;
      }
      if (ii_src_2 >= y_m - 1 && ii_src_3 >= x_m - 1 && ii_src_2 <= y_M + 1 && ii_src_3 <= x_M + 1)
      {
        float r9 = 2.82239990787506e-2F*px*py*(vp[ii_src_3 + 2][ii_src_2 + 2]*vp[ii_src_3 + 2][ii_src_2 + 2])*src[time][p_src];
        u[t1][ii_src_3 + 2][ii_src_2 + 2] += r9;
      }
    }
    /* End section1 */
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec-start_section1.tv_sec)+(double)(end_section1.tv_usec-start_section1.tv_usec)/1000000;
    struct timeval start_section2, end_section2;
    gettimeofday(&start_section2, NULL);
    /* Begin section2 */
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int abc_y_left = y_m; abc_y_left <= abc_y_left_ltkn + y_m - 1; abc_y_left += 1)
      {
        u[t1][x + 2][abc_y_left + 2] = -u[t1][x + 2][42 - abc_y_left];
      }
    }
    /* End section2 */
    gettimeofday(&end_section2, NULL);
    timers->section2 += (double)(end_section2.tv_sec-start_section2.tv_sec)+(double)(end_section2.tv_usec-start_section2.tv_usec)/1000000;
    if ((time)%(12) == 0)
    {
      struct timeval start_section3, end_section3;
      gettimeofday(&start_section3, NULL);
      /* Begin section3 */
      for (int x = x_m; x <= x_M; x += 1)
      {
        #pragma omp simd aligned(u,usave:32)
        for (int y = y_m; y <= y_M; y += 1)
        {
          usave[time / 12][x + 2][y + 2] = u[t0][x + 2][y + 2];
        }
      }
      /* End section3 */
      gettimeofday(&end_section3, NULL);
      timers->section3 += (double)(end_section3.tv_sec-start_section3.tv_sec)+(double)(end_section3.tv_usec-start_section3.tv_usec)/1000000;
    }
    struct timeval start_section4, end_section4;
    gettimeofday(&start_section4, NULL);
    /* Begin section4 */
    for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
    {
      float r11 = (int)(floor(-2.0e-1F*o_y + 2.0e-1F*rec_coords[p_rec][1]));
      float r10 = (int)(floor(-2.0e-1F*o_x + 2.0e-1F*rec_coords[p_rec][0]));
      float sum = 0.0F;
      int ii_rec_1 = r11;
      int ii_rec_2 = r11 + 1;
      float py = (float)(-5.0F*r11 - o_y + rec_coords[p_rec][1]);
      int ii_rec_0 = r10;
      int ii_rec_3 = r10 + 1;
      float px = (float)(-5.0F*r10 - o_x + rec_coords[p_rec][0]);
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1)
      {
        sum += (4.0e-2F*px*py - 2.0e-1F*px - 2.0e-1F*py + 1)*u[t0][ii_rec_0 + 2][ii_rec_1 + 2];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= y_M + 1)
      {
        sum += (-4.0e-2F*px*py + 2.0e-1F*py)*u[t0][ii_rec_0 + 2][ii_rec_2 + 2];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= x_M + 1)
      {
        sum += (-4.0e-2F*px*py + 2.0e-1F*px)*u[t0][ii_rec_3 + 2][ii_rec_1 + 2];
      }
      if (ii_rec_2 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_2 <= y_M + 1 && ii_rec_3 <= x_M + 1)
      {
        sum += 4.0e-2F*px*py*u[t0][ii_rec_3 + 2][ii_rec_2 + 2];
      }
      rec[time][p_rec] = sum;
    }
    /* End section4 */
    gettimeofday(&end_section4, NULL);
    timers->section4 += (double)(end_section4.tv_sec-start_section4.tv_sec)+(double)(end_section4.tv_usec-start_section4.tv_usec)/1000000;
  }
  return 0;
}
