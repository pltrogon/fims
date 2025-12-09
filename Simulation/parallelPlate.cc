/*
 * parallelplate.cc
 * 
 * Garfield++ simulation of a single-electron avalanche.
 * 
 * In a parallel-plate geometry
 */

//Garfield includes
#include "Garfield/ComponentAnalyticField.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/Sensor.hh"

//C includes
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <cstdio>
#include <vector>

using namespace Garfield;

int main(int argc, char* argv[]) {

  if(argc != 6 && argc != 7){
    std::cerr << "Error - Invalid inputs." << std::endl;
    return 1;
  }
  
  const double MICRONTOCM = 1e-4;
  const int electronThreshold = 10;
  const double targetEfficiency = 0.95;
  const double confidenceValue = 2;

  double standoff = std::atof(argv[1]);
  double gap = standoff*MICRONTOCM;
  double eField = std::atof(argv[2]);//kV/cm
  MediumMagboltz* gasFIMS = new MediumMagboltz();

  std::string gasComp = argv[3];
  std::string gasComposition;

  //Gas composition
  if((gasComp == "T2K") || (gasComp == "myT2K")){
    if(argc != 7){
      std::cerr << "Error - Invalid T2K inputs." << std::endl;
      return 1;
    }
    int gasCompAr = std::atoi(argv[4]);
    int gasCompCF4 = std::atoi(argv[5]);
    int gasCompIsobutane = std::atoi(argv[6]);
    int gasSum = gasCompAr + gasCompCF4 + gasCompIsobutane;
    if(abs(gasSum -100) > 1e-3){
      std::cerr << "Error: Gas composition is not 100%." << std::endl;
      return 1;
    }
    gasComposition = gasComp+"-"+std::to_string(gasCompAr)+"-"+std::to_string(gasCompCF4)+"-"+std::to_string(gasCompIsobutane);
    gasFIMS->SetComposition(
      "ar", gasCompAr,
      "cf4", gasCompCF4, 
      "iC4H10", gasCompIsobutane
    );
    gasFIMS->EnablePenningTransfer(0.385, .0, "ar");
  }
  else if(gasComp == "ArCO2"){
    if(argc != 6){
      std::cerr << "Error - Invalid ArCO2 inputs." << std::endl;
      return 1;
    }
    int gasCompAr = std::atoi(argv[4]);
    int gasCompCO2 = std::atoi(argv[5]);
    int gasSum = gasCompAr + gasCompCO2;
    if(abs(gasSum -100) > 1e-3){
      std::cerr << "Error: Gas composition is not 100%." << std::endl;
      return 1;
    }

    gasComposition = gasComp+"-"+std::to_string(gasCompAr)+"-"+std::to_string(gasCompCO2);
    gasFIMS->SetComposition(
      "ar", gasCompAr, 
      "co2", gasCompCO2
    );
    gasFIMS->EnablePenningTransfer(0.51, .0, "ar");
  }
  
  else{
    std::cerr << "Error - Invalid Gas Composition. Options are: T2K, ArCO2, myT2K" << std::endl;
    return 1;
  }

  //gas parameters:
  gasFIMS->SetTemperature(293.15);
  gasFIMS->SetPressure(760.);
  gasFIMS->SetMaxElectronEnergy(200);
  gasFIMS->Initialise(true);

  ComponentAnalyticField parallePlate;
  parallePlate.SetMedium(gasFIMS);


  double voltage = eField*gap*1000;
  parallePlate.AddPlaneY(0., 0.);
  parallePlate.AddPlaneY(gap, -voltage);
  
  parallePlate.PrintCell();


  //Create a sensor
  Sensor* sensorFIMS = new Sensor();
  sensorFIMS->AddComponent(&parallePlate);
  sensorFIMS->SetArea(
    -gap, -gap, -gap, 
    gap, gap, gap
  );

  int avalancheLimit = 20;
  AvalancheMicroscopic* avalancheE = new AvalancheMicroscopic;
  avalancheE->SetSensor(sensorFIMS);
  avalancheE->EnableAvalancheSizeLimit(avalancheLimit);
      
  // ***** Prepare Avalanche Electron ***** //
  //Set the Initial electron parameters
  double x0 = 0., y0 = .99*gap, z0 = 0.;
  double t0 = 0.;//ns
  double e0 = 0.1;//eV (Garfield is weird when this is 0.)
  double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity

  //Set up variables for simulation
  int totalAvalanches = 0;
  int numAboveThreshold = 0;
  int numNoAvalanche = 0;

  double efficiency = 0.;
  double varience = 0.;
  double efficiencyErr = 0.;

  int numInBunch = 100;//Always do at least 100 avalanches first
  double lowerLimit = 0.;
  double upperLimit = 1.;

  //Begin simulating electron avalanches
  bool runAvalanche = true;
  bool isEfficient = false;
  int numAvalanche = 1000;
  while(runAvalanche && totalAvalanches < numAvalanche){

    //Do bunch of avalanches
    for(int inAvalanche = 0; inAvalanche < numInBunch; inAvalanche++){
      
      totalAvalanches++;

      //Begin single-electron avalanche
      avalancheE->AvalancheElectron(x0, y0, z0, 0., e0, dx0, dy0, dz0);

      //Electron count - use endpoints to include attached electrons
      int avalancheElectrons = avalancheE->GetNumberOfElectronEndpoints();

      //Increment stats counters
      if(avalancheElectrons == 1){
        numNoAvalanche++;
      }
      if(avalancheElectrons >= electronThreshold){
        numAboveThreshold++;
      }

    }//end of avalanche bunch loop

		numInBunch = 25;//do bunches of 25 after first iteration

    //Efficiency calculations
    double success = numAboveThreshold;
    double total = totalAvalanches - numNoAvalanche;

    //Binomial Stats
    //efficiency = success/total;
    //varience = (efficiency*(1-efficiency)/total;

    //Bayesian Statistics
    efficiency = (success+1)/(total+2);
    varience = ((success+1)*(success+2))/((total+2)*(total+3)) - efficiency*efficiency;
   
    efficiencyErr = std::sqrt(varience);

    // *** Check efficiency ***
    lowerLimit = efficiency - confidenceValue*efficiencyErr;
    upperLimit = efficiency + confidenceValue*efficiencyErr;

    //Efficiency excludes target within confidence
    if(upperLimit < targetEfficiency){
      runAvalanche = false;
      isEfficient = false;
    }
    
    //Efficiency is above target within confidence
    if(lowerLimit >= targetEfficiency){
      runAvalanche = false;
      isEfficient = true;
    }

		//occasionaly print values
		if(totalAvalanches%100 == 0){
			std::cout << "Total avalanches: " << totalAvalanches << "\n";
      std::cout << "\tEfficiency: " << efficiency << " +/- " << efficiencyErr << "\n";
		}

  }//end gain convergence loop


  std::cout << "--------------------------------------\n";
  std::cout << "Parallel Plate Efficiency:\n";
  std::cout << "\tGap = " << standoff << " um\n";
  std::cout << "\tField = " << eField << " kV/cm\n";
  std::cout << "\tVoltage = " << voltage << " V\n";
  std::cout << "isEfficient = " << isEfficient << "\n";
  std::cout << "Efficiency = " << efficiency << " +/- " << efficiencyErr << "\n";
  std::cout << "totalAvalanches = " << totalAvalanches << "\n";
  std::cout << "--------------------------------------\n";

  return 0;

}
