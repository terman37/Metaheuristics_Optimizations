#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <math.h>

#define abss(a)     (a<0 ? (-a) : a)
#define max(a , b)  (a>b ?   a  : b)
#define min(a , b)  (a>b ?   b  : a)


double func(int func_number , int dim , double* x);
double Shifted_Sphere( int dim , double* x );
double Schwefel_Problem( int dim , double* x );
double Shifted_Rosenbrock( int dim , double* x );
double Shifted_Rastrigin( int dim , double* x );
double Shifted_Griewank( int dim , double* x );
double Shifted_Ackley( int dim , double* x );
double Shifted_Weierstrass( int dim , double* x );

#endif
