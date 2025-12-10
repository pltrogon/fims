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

	int avalancheLimit = 1000;
	AvalancheMicroscopic* avalancheE = new AvalancheMicroscopic;
	avalancheE->SetSensor(sensorFIMS);
	avalancheE->EnableAvalancheSizeLimit(avalancheLimit);
			
	// ***** Prepare Avalanche Electron ***** //
	//Set the Initial electron parameters
	double x0 = 0., y0 = .99*gap, z0 = 0.;
	double t0 = 0.;//ns
	double e0 = 0.1;//eV (Garfield is weird when this is 0.)
	double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity
	
	int numAvalanche = 2000;
	std::vector<int> avalancheElectrons(numAvalanche);
	std::vector<int> avalancheGain(numAvalanche);

	int numHitLimit = 0;
	double xi, yi, zi, ti, Ei;
	double xf, yf, zf, tf, Ef;
	int stat;

	for(int inAvalanche = 0; inAvalanche < numAvalanche; inAvalanche++){

		//Reset avalanche data
		avalancheElectrons[inAvalanche] = 0;
		avalancheGain[inAvalanche] = 0;
		int attachedElectrons = 0;

		//Begin single-electron avalanche
		avalancheE->AvalancheElectron(x0, y0, z0, 0., e0, dx0, dy0, dz0);

		//Electron count - use endpoints to include attached electrons
		int numElectrons = avalancheE->GetNumberOfElectronEndpoints();

		//Check if avalanche limit was reached
		if(numElectrons >= avalancheLimit){
				numHitLimit++;
		}

		//Loop through all electrons in avalanche
		for(int inElectron = 0; inElectron < numElectrons; inElectron++){
				//Extract individual electron data
				avalancheE->GetElectronEndpoint(
					inElectron, 
					xi, yi, zi, ti, Ei, 
					xf, yf, zf, tf, Ef, 
					stat
				);

				if(stat == -7){
					attachedElectrons++;
				}
		}//end electrons in avalanche loop

		avalancheElectrons[inAvalanche] = numElectrons;
		avalancheGain[inAvalanche] = numElectrons - attachedElectrons;

	}//End avalanche loop

	delete avalancheE;


	//***** Output avalanche values *****//	
	//create output file
  std::string dataFilename = "parallelPlateGainFile.txt";
  std::string dataPath = "../../Data/"+dataFilename;
	std::ofstream dataFile;

	//Write results to file
	dataFile.open(dataPath);
  if(!dataFile.is_open()){
    std::cerr << "Error: Could not open file: " << dataPath << std::endl;
  }

	//Output some extra information
	dataFile << "# Parallel Plate Gain\n";
	dataFile << "#\tTotal avalanches = " << numAvalanche << "\n";
	dataFile << "#\tGap = " << standoff << " um\n";
	dataFile << "#\tField = " << eField << " kV/cm\n";
	dataFile << "#\tVoltage = " << voltage << " V\n";
	dataFile << "#\tAvalanche limit = " << avalancheLimit << "\n";
	dataFile << "#\tHit limit = " << numHitLimit << "\n";

	//Output individual avalanche gains
	for(const auto& gainValue : avalancheGain){
		dataFile << gainValue << "\n";
	}

	dataFile.close();

	return 0;

}
