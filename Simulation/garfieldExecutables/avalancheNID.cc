#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

#include <TApplication.h>
#include <TMath.h>
#include <TTree.h>
#include <TFile.h>

#include "Garfield/ComponentElmer.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/Medium.hh"
#include "Garfield/AvalancheMC.hh"
#include "Garfield/Sensor.hh"
#include "Garfield/DriftLineRKF.hh"
#include "Garfield/ViewDrift.hh"
#include "Garfield/ViewSignal.hh"

#include "GarfieldConstants.hh"
#include "Random.hh"
#include "AvalancheNIMicroscopic.hh"

#include "myFunctions.hh"

using namespace Garfield;


int main(int argc, char *argv[]){

	// Handle geometry mode from input
	if(argc != 2){
		std::cerr << "Format: " << argv[0] << " <GeometryMode>" << std::endl;
		return -1;
	}
	std::string geometryModeString = argv[1];
	GeometryMode geometryMode = stringToGeometryMode(argv[1]);
	if(geometryMode == GeometryMode::Unknown){
		std::cerr << "Error: Invalid geometryMode: " << argv[1] << std::endl;
		return -1;
	}

	// Get sensor list
    std::vector<std::string> sensorList;
    sensorList.push_back("CentralPad");
    switch(geometryMode){
		case GeometryMode::Hexagonal: {
			// Add pads to sensor list
			sensorList.push_back("RightTopPad");
			break;
			}

        case GeometryMode::SquareSurrounding:{
            sensorList.push_back("TopPad");
            sensorList.push_back("RightTopPad");
            sensorList.push_back("RightPad");
            sensorList.push_back("RightBottomPad");
            sensorList.push_back("BottomPad");
            sensorList.push_back("LeftBottomPad");
            sensorList.push_back("LeftPad");
            sensorList.push_back("LeftTopPad");
            break;
        }

        case GeometryMode::HexagonalSurrounding:{
            sensorList.push_back("TopPad");
            sensorList.push_back("BottomPad");
            sensorList.push_back("RightTopPad");
            sensorList.push_back("RightBottomPad");
            sensorList.push_back("LeftTopPad");
            sensorList.push_back("LeftBottomPad");
            break;
        }

        default:
            return -1;
    }

	//***** Simulation Parameters *****//
	/*
    auto simParams = readSimulationParameters();
    if(!simParams){
        return -1;
    }
    //Get half of cell scale
    double cellXScale;
    double xScale, yScale;
    switch(geometryMode){
        case GeometryMode::SquareSurrounding:{
            xScale = simParams->pitch/2.;
            yScale = simParams->pitch/2.;
            cellXScale = 0.5;
            break;
        }
        case GeometryMode::HexagonalSurrounding:{
            xScale = simParams->pitch/std::sqrt(3.);
            yScale = simParams->pitch/2.;
            cellXScale = 1./std::sqrt(3.);
            break;
        }
        default:
            return -1;
    }
	*/

	//*************** DATA TREES ***************//
	// ***** Output data file ***** //  
    std::string dataFilename = "NIDData.root";
    std::string dataPath = "../../Data/"+dataFilename;
    TFile *dataFile = new TFile(dataPath.c_str(), "RECREATE");

	// ***** Particle Data tree ***** // 
	TTree *particleDataTree = new TTree("particleDataTree", "Particle Data");
	std::vector<double> xPos, yPos, zPos, time, changePoint;
	int numSteps, exitStatus;

	particleDataTree->Branch("numSteps", &numSteps);
	particleDataTree->Branch("exitStatus", &exitStatus);
	particleDataTree->Branch("xPos", &xPos);
	particleDataTree->Branch("yPos", &yPos);
	particleDataTree->Branch("zPos", &zPos);
	particleDataTree->Branch("time", &time);
	particleDataTree->Branch("changePoint", &changePoint);

	
	std::cout << "Initializing gas mixture..." << std::endl;
    Garfield::MediumMagboltz* gasNID = new Garfield::MediumMagboltz();
    
	// Set STP gas parameters
    double gasTemperature = 293.15;  // K
    double gasPressure = 760.0;       // torr     NOTE: Example is at 76 torr!
    int maxElectronE = 200;           // eV

    gasNID->SetComposition("sf6", 100.0);
	gasNID->SetTemperature(gasTemperature);
	gasNID->SetPressure(gasPressure);
	gasNID->SetMaxElectronEnergy(maxElectronE);
	gasNID->EnableDrift();
	gasNID->Initialise(true);
    
	// Load ion mobilities
	const std::string ionPath = "../NID/IonMobility_SF6-_SF6.txt";
	gasNID->LoadIonMobility(ionPath);

	std::string geometryPath = "../Geometry/";
    std::string elmerResultsPath = geometryPath+"elmerResults/";
    std::string fieldPath = elmerResultsPath + geometryModeString + ".result";
    ComponentElmer fieldFIMS(
        elmerResultsPath+"mesh.header",
        elmerResultsPath+"mesh.elements",
        elmerResultsPath+"mesh.nodes", 
        geometryPath+"dielectrics.dat",
        fieldPath, 
        "mum"
    );

	// Get region of elmer geometry
    double xmin, ymin, zmin, xmax, ymax, zmax;
    fieldFIMS.GetBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax);

    //Define boundary region for simulation
    double xBoundary[2], yBoundary[2], zBoundary[2];
    zBoundary[0] = zmin;
    zBoundary[1] = zmax;
    //Extend simulation boundary to +/- 2*pitch in x and y
	double pitch = 55.;
    xBoundary[0] = -2*pitch;//-2.*simParams->pitch;
    xBoundary[1] = 2*pitch;//2.*simParams->pitch;
    yBoundary[0] = -2*pitch;//-2.*simParams->pitch;
    yBoundary[1] = 2*pitch;//2.*simParams->pitch;

    //Enable periodicity and set components
    fieldFIMS.EnableMirrorPeriodicityX();
    fieldFIMS.EnableMirrorPeriodicityY();
    fieldFIMS.SetGas(gasNID);

	// Import the weighting field for the readout electrodes.
    for(int i = 0; i < sensorList.size(); i++){
        std::string inSensor = sensorList.at(i);
        std::string fieldFilePath = elmerResultsPath + geometryModeString + inSensor + "Weighting.result";
        fieldFIMS.SetWeightingField(fieldFilePath, inSensor);
    }

    // Create a sensor
    Sensor* sensorFIMS = new Sensor();
    sensorFIMS->AddComponent(&fieldFIMS);
    sensorFIMS->SetArea(
        xBoundary[0], yBoundary[0], zBoundary[0], 
        xBoundary[1], yBoundary[1], zBoundary[1]
    );
    for(const auto& inSensor : sensorList){
        sensorFIMS->AddElectrode(&fieldFIMS, inSensor);
    }



	//*************** AVALANCHES ***************//

	AvalancheNIMicroscopic *avalancheNID = new AvalancheNIMicroscopic();
	avalancheNID->SetSensor(sensorFIMS);
	avalancheNID->SetElectronTransportCut(1e-20);
	avalancheNID->SetNegativeIonMass(146);
	avalancheNID->SetDistanceSteps(1e-4); // cm

	//set detachment cross section
	avalancheNID->InputDetachCrossSectionData("../NID/SF6-_F-_Detach.txt");
	avalancheNID->SetDetachModel(0); //0: CrossSection, 1: Threshold


	//Set the Initial electron parameters
	double x0 = 0.0, y0 = 0.0, z0 = 0.01;
	//auto [x0, y0] = randomXYinGeometry(geometryMode, cellLength)
  	//double z0 = simParams->initialZFraction * simParams->driftLength;
	double t0 = 0.;//ns
	double e0 = 0.1;//eV (Garfield is weird when this is 0.)
	double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity

	avalancheNID->AvalancheNegativeIon(x0, y0, z0, 0., e0, dx0, dy0, dz0);
	Int_t numElectrons = avalancheNID->GetNumberOfElectronEndpoints();

	for(int i=0; i<numElectrons; i++){
		//Clear memory
		xPos.clear();
		yPox.clear();
		zPos.clear();
		time.clear();
		changePoint.clear();

		int stat;
		double xi, yi, zi, ti, Ei;
		double xf, yf, zf, tf, Ef;
		avalancheNID->GetElectronEndpoint(i, xi, yi, zi, ti, Ei, xf, yf, zf, tf, Ef, stat);

		unsigned int nStep = avalancheNID->GetNumberOfElectronDriftLinePoints(i);
		exitStatus = stat;
		numSteps  = nStep;
		for(int step=0; step<nStep; step++){
			double x, y, z, t;
			int change;
			avalancheNID->GetElectronDriftLinePoint(x, y, z, t, change, step, i);
			xPos.push_back(x);
			yPos.push_back(y);
			zPos.push_back(z);
			time.push_back(t);
			changePoint.push_back(change);
		}
		particleDataTree->Fill();
	}

	particleDataTree->Write();
	delete particleDataTree;

	dataFile->Close();
    delete dataFile;

	return 0;
}