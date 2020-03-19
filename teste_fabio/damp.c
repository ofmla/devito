#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"

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
} ;


int initdamp(struct dataobj *restrict damp_vec, const float h_x, const float h_y, const int x_M, const int x_m, const int y_M, const int abc_x_l_ltkn, const int abc_x_r_rtkn, const int abc_y_l_ltkn, const int abc_y_r_rtkn, struct profiler * timers, const int y_m)
{
  float (*restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]]) damp_vec->data;
  struct timeval start_section0, end_section0;
  gettimeofday(&start_section0, NULL);
  /* Begin section0 */
  for (int abc_x_l = x_m; abc_x_l <= abc_x_l_ltkn + x_m - 1; abc_x_l += 1)
  {
    for (int y = y_m; y <= y_M; y += 1)
    {
      damp[abc_x_l + 1][y + 1] += (-4.12276274369678e-2F*sin(6.28318530717959F*fabs(5.0e-2F*x_m - 5.0e-2F*abc_x_l + 1.05F)) + 2.5904082296183e-1F*fabs(5.0e-2F*x_m - 5.0e-2F*abc_x_l + 1.05F))/h_x;
    }
  }
  for (int abc_x_r = -abc_x_r_rtkn + x_M + 1; abc_x_r <= x_M; abc_x_r += 1)
  {
    for (int y = y_m; y <= y_M; y += 1)
    {
      damp[abc_x_r + 1][y + 1] += (-4.12276274369678e-2F*sin(6.28318530717959F*fabs(-5.0e-2F*x_M + 5.0e-2F*abc_x_r + 1.05F)) + 2.5904082296183e-1F*fabs(-5.0e-2F*x_M + 5.0e-2F*abc_x_r + 1.05F))/h_x;
    }
  }
  for (int x = x_m; x <= x_M; x += 1)
  {
    for (int abc_y_l = y_m; abc_y_l <= abc_y_l_ltkn + y_m - 1; abc_y_l += 1)
    {
      damp[x + 1][abc_y_l + 1] += damp[x + 1][41 - abc_y_l];
    }
    for (int abc_y_r = -abc_y_r_rtkn + y_M + 1; abc_y_r <= y_M; abc_y_r += 1)
    {
      damp[x + 1][abc_y_r + 1] += (-4.12276274369678e-2F*sin(6.28318530717959F*fabs(-5.0e-2F*y_M + 5.0e-2F*abc_y_r + 1.05F)) + 2.5904082296183e-1F*fabs(-5.0e-2F*y_M + 5.0e-2F*abc_y_r + 1.05F))/h_y;
    }
  }
  /* End section0 */
  gettimeofday(&end_section0, NULL);
  timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
  return 0;
}

