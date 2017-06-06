#ifndef SARSA_AGENT
#define SARSA_AGENT

#include "PolicyAgent.h"
#include "FuncApprox.h"

class SarsaAgent:public PolicyAgent{

 private:

  int episodeNumber;
  double lastState[MAX_STATE_VARS];
  int lastAction;
  double lastReward;

 public:

  SarsaAgent(int numFeatures, int numActions, double learningRate, double epsilon, FunctionApproximator *FA, char *loadWeightsFile, char *saveWeightsFile);
  ~SarsaAgent();

  int  argmaxQ(double state[]);
  int  argmaxQ(double state[], double& bestValueRet);
  double computeQ(double state[], int action);


  int selectAction(double state[], double& bestValueRet);
  int selectAction(double state[]);

  void update(double state[], int action, double reward, double discountFactor);
  void endEpisode();
  void reset();

};

#endif

