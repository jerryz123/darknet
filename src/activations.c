#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}
void leaky_activate_v(float* x, const int n)
{
  setvcfg(0, 1, 0, 2);
  asm volatile ("vmcs vs2, %0"
                :
                : "r" (0.1f));

  for (int i = 0; i < n; )
    {
      int consumed = setvlen(n - i);
      asm volatile ("vmca va0, %0"
                    :
                    : "r" (&x[i]));
      asm volatile ("la t0, vleaky_activate"
                    :
                    :
                    : "t0");
        asm volatile ("lw t1, 0(t0)");
        asm volatile ("vf 0(t0)");
        i += consumed;
    }
  asm volatile ("fence");

}

void leaky_activate_vh(int16_t* x, const int n)
{
  float f;
  setvcfg(0, 0, 1, 2);
  float a = 0.1f;
  asm volatile ("vmcs vs2, %0"
                :
                : "r" (a));

  for (int i = 0; i < n; )
    {
      int consumed = setvlen(n - i);
      asm volatile ("vmca va0, %0"
                    :
                    : "r" (&x[i]));
      asm volatile ("la t0, vleaky_activate_h"
                    :
                    :
                    : "t0");
        asm volatile ("lw t1, 0(t0)");
        asm volatile ("vf 0(t0)");
        i += consumed;
    }
  asm volatile ("fence");

}
void logistic_activate_h(int16_t* x, const int n)
{
  float buf[1024];
  for (int i = 0; i < n; )
    {
      int consumed = 1024 < n - i ? 1024 : n - i;
      cvt_single_prec (&x[i], buf, consumed);
      for (int j = 0; j < consumed; j++)
        buf[j] = logistic_activate (buf[j]);
      cvt_half_prec (buf, &x[i], consumed);
      i += consumed;
    }
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
  if (a == LEAKY)
    {
      leaky_activate_v(x, n);
    }
  else if (a == LINEAR)
    {
      return;
    }
  else
    {
      int i;
      for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
      }
    }
}
void activate_array_h(int16_t *x, const int n, const ACTIVATION a)
{
  if (a == LEAKY)
    {
      leaky_activate_vh(x, n);
    }
  else if (a == LINEAR)
    {
      return;
    }
  else if (a == LOGISTIC)
    {
      logistic_activate_h(x, n);
    }

}

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
}
