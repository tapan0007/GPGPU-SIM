#include "SarsaAgent.h"

using namespace std;

SarsaAgent::SarsaAgent(int numFeatures, int numActions, double learningRate, double epsilon, FunctionApproximator *FA, char *loadWeightsFile, char *saveWeightsFile):PolicyAgent(numFeatures, numActions, learningRate, epsilon, FA, loadWeightsFile, saveWeightsFile){

  episodeNumber = 0;
  lastAction = -1;

}

void SarsaAgent::update(double state[], int action, double reward, double discountFactor){

  if(lastAction == -1){

    for(int i = 0; i < getNumFeatures(); i++){
      lastState[i] = state[i];
    }
    lastAction = action;
    lastReward = reward;
  }
  else{

    FA->setState(lastState);

    double oldQ = FA->computeQ(lastAction);
    FA->updateTraces(lastAction);

    double delta = lastReward - oldQ;

    FA->setState(state);

    //Sarsa update
    double newQ = FA->computeQ(action);

    //    cout << "Update Q: " << newQ << "\n";

    delta += discountFactor * newQ;

    FA->updateWeights(delta, learningRate);
    FA->decayTraces(0);//Assume gamma, lambda are 0.

    for(int i = 0; i < getNumFeatures(); i++){
      lastState[i] = state[i];
    }
    lastAction = action;
    lastReward = reward;
  }
}

void SarsaAgent::endEpisode(){

  episodeNumber++;

  if(lastAction == -1){
    return;//This will not happen usually during keepaway, but is a safety.
  }
  else{

    FA->setState(lastState);
    double oldQ = FA->computeQ(lastAction);
    FA->updateTraces(lastAction);
    double delta = lastReward - oldQ;

    //    cout << "Last Q: " << lastReward << "\n";

    FA->updateWeights(delta, learningRate);
    FA->decayTraces(0);//Assume lambda is 0.
  }

  if(toSaveWeights && (episodeNumber + 1) % 5 == 0){
    saveWeights(saveWeightsFile);
    cout << "Saving weights to " << saveWeightsFile << "\n";
  }

  lastAction = -1;

}

void SarsaAgent::reset(){
  
  lastAction = -1;
}

unsigned int gRandCnt = 0;
unsigned int gMaxQCnt = 0;
int SarsaAgent::selectAction(double state[]){

  int action;

  //double randVal = drand48();
  long int randVal = random();
  float rVal = (randVal / (float) RAND_MAX);

  if(rVal < epsilon){
    //action = (int)(randVal * getNumActions()) % getNumActions();
    action = randVal % getNumActions();
	gRandCnt++;
  }
  else{
    action = argmaxQ(state);
	gMaxQCnt++;
  }
  if (((gRandCnt + gMaxQCnt) % 30000) == 0)
  	printf("random actions %u, max Q actions %u\n", gRandCnt, gMaxQCnt);
  
  return action;
}

int SarsaAgent::selectAction(double state[], double& bestValueRet){

  int action;

  //double randVal = drand48();
  long int randVal = random();
  float rVal = (randVal / (float) RAND_MAX);

  if(rVal < epsilon){
    //action = (int)(randVal * getNumActions()) % getNumActions();
    action = randVal % getNumActions();
	gRandCnt++;
  }
  else{
    action = argmaxQ(state, bestValueRet);
	gMaxQCnt++;
  }
  if (((gRandCnt + gMaxQCnt) % 30000) == 0)
  	printf("random actions %u, max Q actions %u\n", gRandCnt, gMaxQCnt);
  
  return action;
}

extern bool gPrintQvalues;

int SarsaAgent::argmaxQ(double state[])
{
  double Q[getNumActions()];

  FA->setState(state);

  for(int i = 0; i < getNumActions(); i++){
    Q[i] = FA->computeQ(i);
  }
  
  int bestAction = 0;
  double bestValue = Q[bestAction];

  int numTies = 0;

  double EPS=1.0e-4;

  for (int a = 1; a < getNumActions(); a++){

    double value = Q[a];
    if(fabs(value - bestValue) < EPS){
      numTies++;
      
      if(drand48() < (1.0 / (numTies + 1))){
	    bestValue = value;
	    bestAction = a;
      }
    }
    else if (value > bestValue){
      bestValue = value;
      bestAction = a;
      numTies = 0;
    }
  }
  
  return bestAction;
}

int SarsaAgent::argmaxQ(double state[], double& bestValueRet)
{
  double Q[getNumActions()];

  FA->setState(state);

  for(int i = 0; i < getNumActions(); i++){
    Q[i] = FA->computeQ(i);
  }
  
  int bestAction = 0;
  double bestValue = Q[bestAction];
  int numTies = 0;

  if (gPrintQvalues)
	printf("q value = %f\n", bestValue);

  double EPS=1.0e-4;

  for (int a = 1; a < getNumActions(); a++){

    double value = Q[a];
    if(fabs(value - bestValue) < EPS){
      numTies++;
      
      if(drand48() < (1.0 / (numTies + 1))){
	    bestValue = value;
	    bestAction = a;
      }
    }
    else if (value > bestValue){
      bestValue = value;
	  if (gPrintQvalues)
		printf("q value = %f\n", value);
      bestAction = a;
      numTies = 0;
    }
  }
  bestValueRet = bestValue;
  
  return bestAction;
}

double SarsaAgent::computeQ(double state[], int action){//Be careful--this resets FA->state

  FA->setState(state);
  double QValue = FA->computeQ(action);

  return QValue;
}

