/*
 * checkEfficiency.cc
 *
 * 
 * TODO
 * 
 */

// My includes
#include "myFunctions.h"

//Garfield includes
#include "Garfield/ComponentElmer.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/Medium.hh"
#include "Garfield/AvalancheMC.hh"
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
#include <random>
#include <utility>

using namespace Garfield;

int main(int argc, char * argv[]) {
    if(argc != 4){
        std::cerr << "Format: " << argv[0] << " <GeometryMode> <EfficiencyMode> <Target Value> <Detection Threshold>" << std::endl;
        return -1;
    }
    
    std::string geoModeString = argv[1];
    GeometryMode geometryMode = stringToGeometryMode(argv[1]);
    if(geometryMode == GeometryMode::Unknown){
        std::cerr << "Error: Invalid GeometryMode: " << argv[1] << std::endl;
        return -1;
    }

    std::string effModeString = argv[2];
    EfficiencyMode effMode = stringToEfficiencyMode(argv[2]);
    if(effMode == EfficiencyMode::Unknown){
        std::cerr << "Error: Invalid EfficiencyMode: " << argv[2] << std::endl;
        return -1;
    }

    double targetEfficiency = std::atof(argv[3]);
    int electronThreshold = std::atoi(argv[4]);

    //Randon seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    const double MICRONTOCM = 1e-4;

    //Read in simulation parameters
    auto simParams = readSimulationParameters();
    if(!simParams){
        return -1;
    }

    //********** Setup Simulation **********//
    std::cout << "Getting efficiency for " << simParams->runNumber << "(" << effModeString << ")" << std::endl;

    //Gas Mixture
    MediumMagboltz* gasFIMS = initializeGas(*simParams);
    //Field map
    std::string geometryPath = "../Geometry/";
    std::string elmerResultsPath = geometryPath+"elmerResults/";
    std::string fieldPath = elmerResultsPath + geoModeString + ".result";
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
    double cellLength = simParams->pitch/std::sqrt(3.);

    double xBoundary[2], yBoundary[2], zBoundary[2];
    zBoundary[0] = zmin;
    zBoundary[1] = zmax;
    xBoundary[0] = -2.*cellLength;
    xBoundary[1] = 2.*cellLength;
    yBoundary[0] = -2.*simParams->pitch;
    yBoundary[1] = 2.*simParams->pitch;

    //Enable periodicity and set components
    fieldFIMS.EnableMirrorPeriodicityX();
    fieldFIMS.EnableMirrorPeriodicityY();
    fieldFIMS.SetGas(gasFIMS);

    //Create a sensor
    Sensor* sensorFIMS = new Sensor();
    sensorFIMS->AddComponent(&fieldFIMS);
    sensorFIMS->SetArea(
        xBoundary[0], yBoundary[0], zBoundary[0], 
        xBoundary[1], yBoundary[1], zBoundary[1]
    );    

    //Define avalanche characteristics
    int electronLimit = electronThreshold+5;
    AvalancheMicroscopic* avalancheE = new AvalancheMicroscopic;
    avalancheE->SetSensor(sensorFIMS);
    avalancheE->EnableAvalancheSizeLimit(electronLimit);
    {
        SilenceCerr guard;
        avalancheE->EnablePlotting(nullptr, 10);//For velocity vector
    }

    //Deafult initial electron parameters
    double x0 = 0., y0 = 0., z0 = 0.75*simParams->cathodeHeight;
    double t0 = 0.;//ns
    double e0 = 0.1;//eV (Garfield is weird when this is 0.)
    double dx0 = 0., dy0 = 0., dz0 = 0.;//No velocity

    //Set up some data variables
    int numInitialElectrons = 0;//Number of initial electrons generated 
    int numTotalTrials = 0;//Number of electrons populated
    int numAboveThreshold = 0;
    int numCollected = 0;
    int numHitGrid = 0;
    int numFailure = 0;

    //Statistics variables
    EfficiencyResults collectionEff;
    EfficiencyResults detectionEff;
    EfficiencyResults netEfficiency;

    //Initial loop control
    int numInBunch = 500;//Always do at least 500 avalanches first
    bool runAvalanche = true, isEfficienct = false;

    std::cout << "Beginning avalanches..." << std::endl;
    //Run avalanches in bunches
    while(runAvalanche && numInitialElectrons < simParams->numAvalanche){
        for(int inAvalanche=0; inAvalanche < numInBunch; inAvalanche++){
            numInitialElectrons++;

            //Random xy on plane
            auto [sampleX, sampleY] = randomXYInHexagon(cellLength);
            double curX = sampleX, curY = sampleY, curZ = z0;
            double curTime = t0;
            double curEnergy = e0;
            double curDx = 0., curDy = 0., curDz = 0.;

            //Parameters to grab electron data
            double xi, yi, zi, ti, Ei;
            double xf, yf, zf, tf, Ef;
            int exitStatus;

            bool repopulate = true;
            while(repopulate){
                //Populate with an electron
                numTotalTrials++;
                {//Guarding against Garfield error. See notes below.
                    SilenceCerr guard;
                
                    avalancheE->AvalancheElectron(
                        curX, curY, curZ, 
                        curTime, curEnergy, 
                        curDx, curDy, curDz
                    );
                }

                int numAvalancheElectrons = avalancheE->GetNumberOfElectronEndpoints();

                //Ensure electron didnt disappear. Reinitialize if so. 
                if(numAvalancheElectrons >= 1){
                    avalancheE->GetElectronEndpoint(0, xi, yi, zi, ti, Ei, xf, yf, zf, tf, Ef, exitStatus);
                }
                else{
                    //std::cerr << "Error: No electrons in avalanche - Restarting." << std::endl;
                    //std::cerr << "\tError at (" << curX << ", " << curY << ", " << curZ << ")" << std::endl;
                    numFailure++;

                    auto [newX, newY] = randomXYInHexagon(cellLength);
                    curX = newX, curY = newY, curZ = z0;
                    curTime = t0;
                    curEnergy = e0;
                    curDx = 0., curDy = 0., curDz = 0.;

                    /**
                     * TODO - Fix this!!
                     * Garfield error: AvalancheMicroscopic::TransportElectrons: Starting point is not in a valid medium.
                     * Tanner notes (13/05/2026)
                     * Still not exactly sure what/why/how this occurs. 
                     * It seesm that when this is happening y = pitch ALWAYS.
                     * But the area is defined to +/- 2*pitch, so it should be fine.
                     * x values seems like they can be anything, but cap at (-cellLength, cellLength)
                     * Again, defined as 2x this, so not sure.
                     * The z values are + and -. Thought maybe they were "in" a hole and translated weird, but must not be the case
                     * Z range is +5, -2 (not hard limits as far as I can tell)
                     * 
                     * The current implementation is that this is just restarting, so although inefficienct it shouldnt affect results
                     */
                }

                //Check if above threshold for detection efficiency. Assume it is collected.
                if(numAvalancheElectrons >= electronThreshold){
                    numAboveThreshold++;
                    numCollected++;
                    break;
                }

                //Only check where electron ends if there is only 1 (Any larger is assumed to be collected)
                if(numAvalancheElectrons == 1){
                    switch(exitStatus){

                        // Electron attached to gas molecule - Restart with initial electron
                        //WARNING - This may cause an infinite loop. Consider max attempts if becomes an issue
                        case -7: {
                            curX = sampleX, curY = sampleY, curZ = z0;
                            curTime = t0;
                            curEnergy = e0;
                            curDx = 0., curDy = 0., curDz = 0.;
                            break;
                        }

                        // Electron leaves drift medium (Hits Grid/Pad/Dielectric)
                        case -5: {
                            if(zf < -1.*simParams->gridThickness){
                                numCollected++;
                            }
                            else{
                                numHitGrid++;
                            }
                            repopulate = false;
                            break;
                        }

                        // Electron leaves the simulation volume - Shift it back
                        //Determine which boundary was hit and shift by pitch to opposite side
                        case -1: {
                            //Shift x or y
                            curX = std::abs(xf) >= cellLength ? -1.*std::copysign(cellLength, xf) : xf;
                            curY = std::abs(yf) >= simParams->pitch ? -1.*std::copysign(simParams->pitch, yf) : yf;
                            curZ = zf;

                            //Get direction vector from 2nd-last and last points along drift
                            double xPrev, yPrev, zPrev, tPrev, xFinal, yFinal, zFinal, tFinal;
                            int nPoints = avalancheE->GetNumberOfElectronDriftLinePoints(0);
                            avalancheE->GetElectronDriftLinePoint(xFinal, yFinal, zFinal, tFinal, nPoints-1, 0);
                            avalancheE->GetElectronDriftLinePoint(xPrev, yPrev, zPrev, tPrev, nPoints-2, 0);

                            //Get normalized direction vector
                            double dx = xFinal - xPrev;
                            double dy = yFinal - yPrev;
                            double dz = zFinal - zPrev;
                            double vMag = std::sqrt(dx*dx + dy*dy + dz*dz);
                            curDx = dx/vMag;
                            curDy = dy/vMag;
                            curDz = dz/vMag;
                            curTime = tf;
                            curEnergy = Ef;
                            break;
                        }

                        default:
                            std::cerr << "Error: Unexpected electron endpoint status (" << exitStatus << ")" << std::endl;
                            return -1;

                    }//End of electron endpoint switch
                }

            }//End of single avalanche trial

        }//end of avalanche bunch loop
        std::cout << "Done " << numInitialElectrons << " trials." << std::endl;
        std::cerr << "Number of surpressed Garfield errors: " << numFailure << std::endl;

        numInBunch = 100;//do bunches of 100 after first iteration

        // *** Check efficiencies ***
        //Collection
        collectionEff = calculateEfficiencyStats(numCollected, numInitialElectrons);
        //Detection
        detectionEff = calculateEfficiencyStats(numAboveThreshold, numCollected);

        //Net efficiency
        netEfficiency.meanValue = collectionEff.meanValue*detectionEff.meanValue;
        netEfficiency.minValue = collectionEff.minValue*detectionEff.minValue;
        netEfficiency.maxValue = collectionEff.maxValue*detectionEff.maxValue;

        netEfficiency.lowError = netEfficiency.meanValue - netEfficiency.minValue;
        netEfficiency.highError = netEfficiency.maxValue - netEfficiency.meanValue;


        //Determine if target efficiency exceeds or excludes target value within confidence
        EfficiencyResults* activeEff;
        switch(effMode){
            case EfficiencyMode::Net:
                activeEff = &netEfficiency; break;
            case EfficiencyMode::Detection:
                activeEff = &detectionEff; break;
            case EfficiencyMode::Collection:
                activeEff = &collectionEff; break;
            default:
                return -1;
        }

        if(activeEff->maxValue < targetEfficiency || activeEff->minValue >= targetEfficiency){
            runAvalanche = false;
        }
    }//End of all avalanches


    //***** Output efficiency value *****//	
    //create output file
    std::string dataFilename = "efficiencyResults.dat";
    std::string dataPath = "../../Data/"+dataFilename;
    std::ofstream dataFile;

    //Write results to file
    dataFile.open(dataPath);
    if(!dataFile.is_open()){
        std::cerr << "Error: Could not open file: " << dataPath << std::endl;
    }

    //Write some general info
    dataFile << "Finding efficiency for run: " << simParams->runNumber << "\n";
    dataFile << "Run mode: " << effModeString << "\n";
    dataFile << "Total initial electrons: " << numInitialElectrons << " (of " << simParams->numAvalanche << ")\n";
    dataFile << "Electron threshold: " << electronThreshold << "\n";
    dataFile << "Field Ratio: " << simParams->fieldRatio << "\n\n";

    //Include the actual results
    dataFile << "Num collected:\n" << numCollected << "\n";
    dataFile << "Num detected:\n" << numAboveThreshold << "\n\n";

    dataFile << "Stop condition:\n";
    if(runAvalanche){
        dataFile << "DID NOT CONVERGE\n\n";
    }
    else{
        dataFile << "CONVERGED\n\n";
    }

    //Include all of the calculated values
    dataFile << "Efficiency Values (Form: meanValue, lowError, highError):\n";
    dataFile << "Collection Efficiency:\n" << collectionEff.meanValue << "\n";
    dataFile << collectionEff.lowError << "\n" << collectionEff.highError << "\n";
    dataFile << "Detection Efficiency:\n" << detectionEff.meanValue << "\n";
    dataFile << detectionEff.lowError << "\n" << detectionEff.highError << "\n";
    dataFile << "Net Efficiency:\n" << netEfficiency.meanValue << "\n";
    dataFile << netEfficiency.lowError << "\n" << netEfficiency.highError << std::endl;

    dataFile.close();

    return 0;
}
