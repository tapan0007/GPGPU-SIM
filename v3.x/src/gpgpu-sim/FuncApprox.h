#ifndef FUNC_APPROX
#define FUNC_APPROX

#include <stdlib.h>
#include <math.h>

#define MAX_STATE_VARS 100
#define MAX_ACTIONS 10

class FunctionApproximator{

 protected:

  int numFeatures, numActions;
  
  /*
    double ranges      [ MAX_STATE_VARS ];
    double minValues   [ MAX_STATE_VARS ];
    double resolutions [ MAX_STATE_VARS ];
    
    double mean[MAX_STATE_VARS];
    double variance[MAX_STATE_VARS];
  */
  
  double state[MAX_STATE_VARS];
  
  int getNumFeatures();
  int getNumActions();
  
  /*
    double getRange     ( int i ) { return ranges     [ i ]; }
    double getMinValue  ( int i ) { return minValues  [ i ]; }
    double getResolution( int i ) { return resolutions[ i ]; }
    double getMean(int i){return mean[i];}
    double getVariance(int i){return variance[i];}
  */
  
 public:
  //  FunctionApproximator( int numF, int numA, 
  //			double r[], double m[], double res[], double mu[], double sigmaSquared[]);
  FunctionApproximator(int numF, int numA);
  virtual ~FunctionApproximator(){}
  

  virtual void setState(double s[]);

  virtual double computeQ(int action) = 0;
  virtual int argMaxQ();
  virtual double bestQ();
  virtual void updateWeights(double delta, double alpha) = 0;

  virtual void clearTraces(int action) = 0;
  virtual void decayTraces(double decayRate) = 0;
  virtual void updateTraces(int action) = 0;

  virtual void read (char *fileName) = 0;
  virtual void write(char *fileName) = 0;

  virtual int getNumWeights() = 0;
  virtual void getWeights(double w[]) = 0;
  virtual void setWeights(double w[]) = 0;

  virtual void reset() = 0;

};

#endif

