/**
 * 
 * TODO
 */

//My includes
#include "myFunctions.hh"

//Garfield includes
#include "Garfield/ComponentElmer.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/Medium.hh"
#include "Garfield/AvalancheMC.hh"
#include "Garfield/Sensor.hh"
#include "Garfield/DriftLineRKF.hh"
#include "Garfield/ViewDrift.hh"
#include "Garfield/ViewSignal.hh"
#include "Garfield/ViewField.hh"

//ROOT includes
#include "TApplication.h"
#include "TTree.h"
#include "TFile.h"
#include "TString.h"
#include "TChain.h"
#include <TH1D.h>
#include <TCanvas.h>

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

int main(int argc, char * argv[]) {

    const double MICRON = 1e-6;
    const double CM = 1e-2;
    const double MICRONTOCM = 1e-4;
    const double ELEMENTARY_CHARGE = 1.602176634e-19;

    if(argc != 2){
        std::cerr << "Format: " << argv[0] << " <GeometryMode>" << std::endl;
        return -1;
    }

    // Handle geometry mode from input
    std::string geometryModeString = argv[1];
    GeometryMode geometryMode = stringToGeometryMode(argv[1]);
    if(geometryMode == GeometryMode::Unknown){
        std::cerr << "Error: Invalid geometryMode: " << argv[1] << std::endl;
        return -1;
    }

    //Random seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Get sensor list
    std::vector<std::string> sensorList;
    sensorList.push_back("CentralPad");
    switch(geometryMode){
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
    // TEST: scale up to unit cell size
    xScale *= 1.5;
    yScale *= 2.0;

    std::cout << "Running avalanche animations... " << std::endl;

    //*************** DATA TREES ***************//
    // ***** Output data file ***** //  
    std::string dataFilename = "animationData.root";
    std::string dataPath = "../../Data/"+dataFilename;
    TFile *dataFile = new TFile(dataPath.c_str(), "RECREATE");

    // ***** Simulation Info tree ***** //
    TTree *simDataTree = new TTree("simDataTree", "Simulation Data");
    double amplificationField = simParams->driftField*simParams->fieldRatio;

    // Geometry parameters
    simDataTree->Branch("padLength", &simParams->padLength);
    simDataTree->Branch("pitch", &simParams->pitch);
    simDataTree->Branch("amplificationGap", &simParams->amplificationGap);
    simDataTree->Branch("gridThickness", &simParams->gridThickness);
    simDataTree->Branch("padThickness", &simParams->padThickness);
    simDataTree->Branch("holeRadius", &simParams->holeRadius);
    simDataTree->Branch("driftLength", &simParams->driftLength);
    simDataTree->Branch("thicknessSiO2", &simParams->thicknessSiO2);
    simDataTree->Branch("pillarRadius", &simParams->pillarRadius);

    // Field parameters
    simDataTree->Branch("driftField", &simParams->driftField);
    simDataTree->Branch("amplificationField", &amplificationField);
    simDataTree->Branch("fieldRatio", &simParams->fieldRatio);
    simDataTree->Branch("numFieldLine", &simParams->numFieldLine);

    // Simulation parameters
    simDataTree->Branch("runNumber", &simParams->runNumber);
    simDataTree->Branch("numAvalanche", &simParams->numAvalanche);
    simDataTree->Branch("avalancheLimit", &simParams->avalancheLimit);
    simDataTree->Branch("numInputs", &simParams->numInputs);
    simDataTree->Branch("initialZFraction", &simParams->initialZFraction);

    // Gas parameters
    simDataTree->Branch("gasCompAr", &simParams->gasCompAr);
    simDataTree->Branch("gasCompCO2", &simParams->gasCompCO2);
    simDataTree->Branch("gasCompCF4", &simParams->gasCompCF4);
    simDataTree->Branch("gasCompIsobutane", &simParams->gasCompIsobutane);
    simDataTree->Branch("gasPenning", &simParams->gasPenning);

    simDataTree->Fill();
    simDataTree->Write();
    delete simDataTree;

    // ***** Avalanche Info tree ***** //
    TTree *avalancheTree = new TTree("avalancheDataTree", "Avalanche Data");
    int avalancheID, gain, numPosIons, numNegIons;

    avalancheTree->Branch("AvalancheID", &avalancheID);
    avalancheTree->Branch("Gain", &gain);
    avalancheTree->Branch("numPosIons", &numPosIons);
    avalancheTree->Branch("numNegIons", &numNegIons);

    // ***** Field tree ***** //
    TTree *fieldTree = new TTree("fieldTree", "Electric and Weighting Field Map");
    double xField, yField, zField;
    double eFieldX, eFieldY, eFieldZ;
    std::vector<double> padWeights(sensorList.size(), 0.0);

    fieldTree->Branch("x", &xField);
    fieldTree->Branch("y", &yField);
    fieldTree->Branch("z", &zField);
    fieldTree->Branch("Ex", &eFieldX);
    fieldTree->Branch("Ey", &eFieldY);
    fieldTree->Branch("Ez", &eFieldZ);
    for (size_t i = 0; i < sensorList.size(); i++) {
        fieldTree->Branch(Form("Weight_%s", sensorList[i].c_str()), &padWeights[i]);
    }

    //***** Field Line Data Tree *****//
    TTree *fieldLineTree = new TTree("fieldLineTree", "Field Lines");
    int fieldLineID, fieldLineStart;
    double fieldLineX, fieldLineY, fieldLineZ;

    fieldLineTree->Branch("FieldLineID", &fieldLineID);
    fieldLineTree->Branch("FieldStart", &fieldLineStart);
    fieldLineTree->Branch("x", &fieldLineX);
    fieldLineTree->Branch("y", &fieldLineY);
    fieldLineTree->Branch("z", &fieldLineZ);

    // ***** Particle Data tree ***** // 
    TTree *animationTree = new TTree("animationDataTree", "3D Particle Data");
    int frameID;
    double frameTime;
    std::vector<int> particleType; // 0 if electron, 1 if +ion, -1 if negative ion
    std::vector<double> xParticle, yParticle, zParticle;

    animationTree->Branch("AvalancheID", &avalancheID);
    animationTree->Branch("FrameID", &frameID);
    animationTree->Branch("Time", &frameTime);
    animationTree->Branch("ParticleType", &particleType);
    animationTree->Branch("x", &xParticle);
    animationTree->Branch("y", &yParticle);
    animationTree->Branch("z", &zParticle);

    // ***** Signal tree ***** //
    TTree* signalDataTree = new TTree("signalDataTree", "Induced Signal");
    double signalTime;
    std::vector<double> padSignals(sensorList.size(), 0.0);
    std::vector<double> electronSignals(sensorList.size(), 0.0);
    std::vector<double> ionSignals(sensorList.size(), 0.0);

    signalDataTree->Branch("AvalancheID", &avalancheID);
    signalDataTree->Branch("Time", &signalTime);
    for (size_t i = 0; i < sensorList.size(); i++) {
        signalDataTree->Branch(Form("Signal_%s", sensorList[i].c_str()), &padSignals[i]);
        signalDataTree->Branch(Form("Electron_%s", sensorList[i].c_str()), &electronSignals[i]);
        signalDataTree->Branch(Form("Ion_%s", sensorList[i].c_str()), &ionSignals[i]);
    }

    //*************** SIMULATION ***************//

    // Define and initialize the gas mixture
    MediumMagboltz* gasFIMS = initializeGas(*simParams); 

    //Import field-map
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
    xBoundary[0] = -2.*simParams->pitch;
    xBoundary[1] = 2.*simParams->pitch;
    yBoundary[0] = -2.*simParams->pitch;
    yBoundary[1] = 2.*simParams->pitch;

    //Enable periodicity and set components
    fieldFIMS.EnableMirrorPeriodicityX();
    fieldFIMS.EnableMirrorPeriodicityY();
    fieldFIMS.SetGas(gasFIMS);

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

    //Set timing for signals
    double signalFinal = 500.;//ns
    double signalStep = 0.01;//ns
    int numSignalBins = signalFinal/signalStep;
    sensorFIMS->SetTimeWindow(0.0, signalStep, numSignalBins);


    //*************** FIELDS ***************//
    std::cout << "Getting electric and weighting field information...\n";

    const int numBins = 101;
    const double nStep = static_cast<double>(numBins - 1);
    double dx = 2.*xScale/nStep;
    double dy = 2.*yScale/nStep;
    double zScale = .99;
    double dz = zScale*(zmax - zmin)/nStep;
    Medium* inMedium;
    int status;

    for(int k=0; k<numBins; k++){
        zField = zScale*zmin + k*dz;
                
        //Do x=0 plane
        for (int j = 0; j < numBins; j++) {
            xField = 0.0;
            yField = -yScale + j*dy;

            fieldFIMS.ElectricField(
                xField, yField, zField, 
                eFieldX, eFieldY, eFieldZ, 
                inMedium, status
            );

            for(size_t i = 0; i < sensorList.size(); i++){
                const char* inPad = sensorList[i].c_str();
                padWeights[i] = fieldFIMS.WeightingPotential(
                    xField, yField, zField, 
                    inPad
                );
            }
            fieldTree->Fill();
        }

        //Do y=0 plane
        for (int i = 0; i < numBins; i++) {
            xField = -xScale + i*dx;
            yField = 0.0;

            fieldFIMS.ElectricField(
                xField, yField, zField, 
                eFieldX, eFieldY, eFieldZ, 
                inMedium, status
            );

            for(size_t i = 0; i < sensorList.size(); i++){
                const char* inPad = sensorList[i].c_str();
                padWeights[i] = fieldFIMS.WeightingPotential(
                    xField, yField, zField, 
                    inPad
                );
            }
            fieldTree->Fill();
        }

    }
    fieldTree->Write();
    delete fieldTree;
    

    //*************** FIELD LINES ***************//
    std::cout << "Generating field lines...\n";

    const int numLines = simParams->numFieldLine;
    const double fieldLineStep = static_cast<double>(numLines - 1);

    dx = 2.*xScale/fieldLineStep;
    dy = 2.*xScale/fieldLineStep;

    DriftLineRKF driftLines(sensorFIMS);
    driftLines.SetMaximumStepSize(MICRONTOCM);

    std::vector<std::array<float, 3> > fieldLines;

    double holeRadius2 = std::pow(simParams->holeRadius, 2.);
    double gridLineStart= 5.*simParams->gridThickness;

    std::pair<int, double> lineLocs[3] = {
        { 0, zmax*zScale},            // Cathode
        {-1, -gridLineStart},       // Below grid
        { 1,  gridLineStart}        // Above grid
    };
    
    // Do all field lines
    for (const auto& inLineLoc : lineLocs) {

        fieldLineStart = inLineLoc.first;
        double zPos = inLineLoc.second;
        bool isGrid = (fieldLineStart != 0);
    
        for(int i=0; i<numLines; i++){
            for(int j=0; j<numLines; j++){

                fieldLineID = i*numLines + j;

                double xPos = -xScale + i*dx;
                double yPos = -yScale + j*dy;

                // Skip any grid lines that are within the hole
                if(isGrid && (xPos*xPos + yPos*yPos < holeRadius2)){
                    continue;
                }
                
                fieldLines.clear();
                driftLines.FieldLine(xPos, yPos, zPos, fieldLines);

                // Get coordinates of every point along field line and fill the tree
                for(const auto& inLine : fieldLines){
                    fieldLineX = inLine[0];
                    fieldLineY = inLine[1];
                    fieldLineZ = inLine[2];

                    fieldLineTree->Fill();
                }
            }
        }
    }
    fieldLineTree->Write();
    delete fieldLineTree;
    
    
    //*************** AVALANCHES ***************//

    int numAvalanche = simParams->numAvalanche;
    std::cout << "Running " << numAvalanche << " avalanches...\n";

    //Set the Initial electron parameters
    double z0 = simParams->initialZFraction * simParams->driftLength;
    double t0 = 0.;//ns
    double e0 = 0.1;//eV (Garfield is weird when this is 0.)
    double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity

    // ***** Single-Electron Avalanche ***** //
    for(int inAvalanche=0; inAvalanche<numAvalanche; inAvalanche++){
        avalancheID = inAvalanche;
        std::cerr << "Beginning avalanche: " << inAvalanche << std::endl;
        //std::cout << "Beginning avalanche: " << inAvalanche << std::endl;

        AvalancheMicroscopic electronAvalanche(sensorFIMS);
        AvalancheMC ionDrift(sensorFIMS);
        electronAvalanche.EnableAvalancheSizeLimit(simParams->avalancheLimit);
        
        double cellLength = simParams->pitch*cellXScale;
        auto [x0, y0] = randomXYinGeometry(geometryMode, cellLength);

        //For tracking particles processed
        std::unordered_set<size_t> electronsWithIonsSpawned;
        std::unordered_set<size_t> attachedElectronsProcessed;
        int totalElectrons = 1;
        int totalPosIons = 0;
        int totalNegIons = 0;

        double tFrameStart = 0., dt = 0.;
        frameID = 0;

        //Initial electron
        electronAvalanche.AddElectron(x0, y0, z0, t0, e0);
            
        while(true){
            //Clear memory
            particleType.clear();
            xParticle.clear();
            yParticle.clear();
            zParticle.clear();

            if(frameID%100==0){
                std::cout << "\tProcessing frame: " << inAvalanche << "." << frameID<< std::endl;
            }

            tFrameStart += dt;

            if(tFrameStart > 100e3){
                std::cout << "***** TIMEOUT *****" << std::endl;
                break;
            }

            //Check if any active electrons or ions exist
            bool activeElectrons = false;
            for(const auto& inElectron : electronAvalanche.GetElectrons()){
                if(!inElectron.path.empty() && inElectron.path.back().t >= tFrameStart){
                    activeElectrons = true;
                    break;
                }
            }

            bool activeIons = false, ionsBelowGrid = false, ionsNearGrid = false;
            for(const auto& inIon : ionDrift.GetIons()){
                if(!inIon.path.empty()){
                    if(inIon.path.back().t >= tFrameStart){activeIons = true;}
                    if(inIon.path.back().z <= 0.0){ionsBelowGrid = true;}
                    if(inIon.path.back().z <= 10.0 * MICRONTOCM){ionsNearGrid = true;}
                }
            }

            bool activeNegIons = false;
            for(const auto& inNegIon : ionDrift.GetNegativeIons()) {
                if(!inNegIon.path.empty() && inNegIon.path.back().t >= tFrameStart){
                    activeNegIons = true;
                    break;
                }
            }

            if(!activeElectrons && !activeIons && !activeNegIons){
                std::cout << "***** NO MORE PARTICLES *****" << std::endl;
                break;
            }
            
            //Determine frame timestep
            if(activeElectrons){dt = 0.01;}
            else if(ionsBelowGrid || activeNegIons){dt = 1.0;}
            else if(ionsNearGrid){dt = 10.0;}
            else{dt = 1000.0;}

            ionDrift.SetTimeSteps(dt/5.);
            frameTime = tFrameStart+dt;
            
            // ***** Process Electrons ***** //
            if(activeElectrons){
                electronAvalanche.SetTimeWindow(tFrameStart, frameTime);
                electronAvalanche.ResumeAvalanche();

                const auto& electrons = electronAvalanche.GetElectrons();

                for(size_t i = 0; i < electrons.size(); i++){
                    const auto& inElectron = electrons[i];
                    if(inElectron.path.empty()){ continue; }

                    // Spawn positive ions for new secondary electrons
                    if(i > 0 && electronsWithIonsSpawned.find(i) == electronsWithIonsSpawned.end()){
                        const double xIon = inElectron.path.front().x;
                        const double yIon = inElectron.path.front().y;
                        const double zIon = inElectron.path.front().z;
                        const double tIon = inElectron.path.front().t;

                        ionDrift.AddIon(xIon, yIon, zIon, tIon);
                        electronsWithIonsSpawned.insert(i);

                        totalElectrons++;
                        totalPosIons++;
                    }

                    // Handle electron attachments -> spawn negative ions
                    if(inElectron.status == -7 && attachedElectronsProcessed.find(i) == attachedElectronsProcessed.end()){
                        const double xAtt = inElectron.path.back().x;
                        const double yAtt = inElectron.path.back().y;
                        const double zAtt = inElectron.path.back().z;
                        const double tAtt = inElectron.path.back().t;

                        ionDrift.AddNegativeIon(xAtt, yAtt, zAtt, tAtt);
                        attachedElectronsProcessed.insert(i);

                        totalNegIons++;
                    }

                    // Animation tracking
                    particleType.push_back(0);
                    xParticle.push_back(inElectron.path.back().x);
                    yParticle.push_back(inElectron.path.back().y);
                    zParticle.push_back(inElectron.path.back().z);
                }
            }

            // ***** Process Ions ***** //
            if(!ionDrift.GetIons().empty() || !ionDrift.GetNegativeIons().empty()){
                ionDrift.SetTimeWindow(tFrameStart, frameTime);
                ionDrift.ResumeAvalanche();

                // Positive Ions
                for(const auto& inIon : ionDrift.GetIons()){
                    if(inIon.path.empty()){ continue; }

                    particleType.push_back(1);
                    xParticle.push_back(inIon.path.back().x);
                    yParticle.push_back(inIon.path.back().y);
                    zParticle.push_back(inIon.path.back().z);
                }

                // Negative Ions
                for(const auto& inNegIon : ionDrift.GetNegativeIons()){
                    if(inNegIon.path.empty()){ continue; }

                    particleType.push_back(-1);
                    xParticle.push_back(inNegIon.path.back().x);
                    yParticle.push_back(inNegIon.path.back().y);
                    zParticle.push_back(inNegIon.path.back().z);
                }
            }

            // Fill animation frame
            animationTree->Fill();
            frameID++;

        }// End of frame loop

        // ********** Avalanche Completed ********** //

        // ***** Avalanche Statistics ***** // 
        gain = totalElectrons;
        numPosIons = totalPosIons;
        numNegIons = totalNegIons;
        avalancheTree->Fill();

        // ***** Induced Signals ***** //
        for(int inSignal=0; inSignal < numSignalBins; inSignal++){
            signalTime = inSignal*signalStep;

            for(size_t i = 0; i < sensorList.size(); i++){
                const char* inPad = sensorList[i].c_str();
                padSignals[i] = sensorFIMS->GetSignal(inPad, inSignal);
                electronSignals[i] = sensorFIMS->GetElectronSignal(inPad, inSignal);
                ionSignals[i] = sensorFIMS->GetIonSignal(inPad, inSignal);
            }

            signalDataTree->Fill();
        }
        sensorFIMS->ClearSignal();
    }
    // ********** All Avalanches Completed ********** //
    

    delete sensorFIMS;
    delete gasFIMS;

    animationTree->Write();
    delete animationTree;
    signalDataTree->Write();
    delete signalDataTree;

    avalancheTree->Write();
    delete avalancheTree;

    dataFile->Close();
    delete dataFile;

    return 0;
}
