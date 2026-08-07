/*
 * checkEffandGain.cc
 * 
 */


// My includes
#include "myFunctions.hh"

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

//Random seed
inline std::mt19937& getRNG(){
    thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

int main(int argc, char * argv[]) {
    if(argc != 2){
        std::cerr << "Format: " << argv[0] << "<GeometryMode>" << std::endl;
        return -1;
    }

    std::string geoModeString = argv[1];
    GeometryMode geometryMode = stringToGeometryMode(argv[1]);
    if(geometryMode == GeometryMode::Unknown){
        std::cerr << "Error: Invalid GeometryMode: " << argv[1] << std::endl;
        return -1;
    }

    const double MICRONTOCM = 1e-4;
    const int minimumThreshold = 10;

    //Read in simulation parameters
    auto simParams = readSimulationParameters();
    if(!simParams){
        return -1;
    }

    //********** Setup Simulation **********//
    std::cout << "Getting efficiency and gain for " << simParams->runNumber << std::endl;

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
    Sensor sensorFIMS;
    sensorFIMS.AddComponent(&fieldFIMS);
    sensorFIMS.SetArea(
        xBoundary[0], yBoundary[0], zBoundary[0], 
        xBoundary[1], yBoundary[1], zBoundary[1]
    );    

    //Object for viewing drift
    viewElectronDrift = new ViewDrift();
    viewElectronDrift->SetArea(
        xBoundary[0], yBoundary[0], zBoundary[0], 
        xBoundary[1], yBoundary[1], zBoundary[1]
    );

    //Define avalanche characteristics
    int electronLimit = simParams->avalancheLimit;
    AvalancheMicroscopic avalancheE;
    avalancheE.SetSensor(&sensorFIMS);
    avalancheE.EnableAvalancheSizeLimit(electronLimit);
    {
        SilenceCerr guard;
        avalancheE.EnablePlotting(viewElectronDrift, 10);//For velocity vector
    }

    //Deafult initial electron parameters
    double x0 = 0., y0 = 0., z0 = simParams->initialZFraction * simParams->driftLength;
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
    int numAttached = 0;
    int numHitLimit = 0;

    std::vector<uint16_t>gains;

    double averageGain = 0.;
    double averageGainVar = 1.;
    double averageGainStdDev = 0.;
    double averageGainErr = 1.;
    double netEffErr = 1.;
    double collEffErr = 1., detectEffErr = 1.;

    //Statistics variables
    EfficiencyResults collectionEff;
    EfficiencyResults detectionEff;
    EfficiencyResults netEfficiency;

    // Avalanche Controls
    const int numInBunch = 500;//Bunches of 500 at a time
    bool runAvalanche = true;

    //Parameters to initialize electrons
    double curX, curY, curZ;
    double curTime, curEnergy;
    double curDx, curDy, curDz;

    //Parameters to grab electron data
    double xi, yi, zi, ti, Ei;
    double xf, yf, zf, tf, Ef;
    int exitStatus;    

    while(runAvalanche && numInitialElectrons < simParams->numAvalanche){
        for(int inAvalanche=0; inAvalanche<numInBunch; inAvalanche++){
            numInitialElectrons++;

            //Reset parameters to initial and random xy on plane
            auto [sampleX, sampleY] = randomXYInHexagon(cellLength);
            curX = sampleX, curY = sampleY, curZ = z0;
            curTime = t0;
            curEnergy = e0;
            curDx = 0., curDy = 0., curDz = 0.;

            //Generate single electron avalanche
            int numAvalancheElectrons = 0;
            bool repopulate = true;

            while(repopulate){
                numTotalTrials++;
                {//Guarding against Garfield error.
                    SilenceCerr guard;
                
                    avalancheE.AvalancheElectron(
                        curX, curY, curZ, 
                        curTime, curEnergy, 
                        curDx, curDy, curDz
                    );
                }

                numAvalancheElectrons = avalancheE.GetNumberOfElectronEndpoints();

                //Electron disappeared - Completely restart with new point
                if(numAvalancheElectrons == 0){
                    //Random xy on plane
                    auto [sampleX, sampleY] = randomXYInHexagon(cellLength);
                    curX = sampleX, curY = sampleY, curZ = z0;
                    curTime = t0;
                    curEnergy = e0;
                    curDx = 0., curDy = 0., curDz = 0.;
                }

                // Only 1 electron - Check how it ended
                else if(numAvalancheElectrons == 1){
                    avalancheE.GetElectronEndpoint(0, xi, yi, zi, ti, Ei, xf, yf, zf, tf, Ef, exitStatus);

                    switch(exitStatus){

                        //Electron attatched. Restart with same initial electron
                        // WARNING - Can inifinite loop here.
                        case -7: {
                            numAttached++;
                            break;
                        }

                        //Electron leave drift medium (Hits grid/Pad/Dielectric)
                        case -5: {
                            repopulate = false;
                            gains.push_back(static_cast<uint16_t>(numAvalancheElectrons));
                            if(zf < -1.*simParams->gridThickness){
                                numCollected++;
                            }
                            else{
                                numHitGrid++;
                            }
                            break;
                        }

                        // Electron leaves simulation volume - Shift it back
                        // Region is 4x wide, but shift to central
                        // Example: x range is -2cell, if x>|cell|, move into cell
                        case -1: {
                            constexpr double eps = 1e-7; // 1 nm nudge to keep inside boundary
                            //Shift x by cellLength if necessary
                            curX = std::abs(xf) >= cellLength ? -1.*std::copysign(cellLength-eps, xf) : xf;
                            //Shift y by pitch if necessary
                            curY = std::abs(yf) >= simParams->pitch ? -1.*std::copysign(simParams->pitch-eps, yf) : yf;
                            curZ = zf;

                            //Get direction vector from 2nd-last and last points along drift
                            double xPrev, yPrev, zPrev, tPrev, xFinal, yFinal, zFinal, tFinal;
                            int nPoints = avalancheE.GetNumberOfElectronDriftLinePoints(0);
                            if(nPoints >= 2){
                                avalancheE.GetElectronDriftLinePoint(xFinal, yFinal, zFinal, tFinal, nPoints-1, 0);
                                avalancheE.GetElectronDriftLinePoint(xPrev, yPrev, zPrev, tPrev, nPoints-2, 0);
                                
                                //Get normalized direction vector
                                double dx = xFinal - xPrev;
                                double dy = yFinal - yPrev;
                                double dz = zFinal - zPrev;
                                double vMag = std::sqrt(dx*dx + dy*dy + dz*dz);

                                if(vMag > 0.){
                                    curDx = dx/vMag;
                                    curDy = dy/vMag;
                                    curDz = dz/vMag;
                                }
                                else{
                                    curDx = 0.; curDy = 0.; curDz = 0.;
                                }
                            }
                            else{
                                curDx = 0.; curDy = 0.; curDz = 0.;
                            }

                            curTime = tf;
                            curEnergy = Ef;
                            break;
                        }

                        default:
                            std::cerr << "Error: Unexpected electron endpoint status (" << exitStatus << ")" << std::endl;
                            return -1;
                    }

                }

                // More than 1 electron
                else{
                    repopulate = false;
                    gains.push_back(static_cast<uint16_t>(numAvalancheElectrons));
                    numCollected++;
                    if(numAvalancheElectrons > minimumThreshold){
                        numAboveThreshold++;
                    }
                    if(numAvalancheElectrons==electronLimit){
                        numHitLimit++;
                    }
                }
                viewElectronDrift->Clear();
            }//end of single-electron avalanche
        }//End of avalanche bunch
        std::cout << "Done " << numInitialElectrons << "trials." << std::endl;

        //Calculate some statistics

        //Gain
        double gainSum = std::accumulate(gains.begin(), gains.end(), 0.0);
        averageGain = gainSum/gains.size();

        double gain2Sum = 0.;
        for(uint16_t g : gains){
            double gainDiff = static_cast<double>(g) - averageGain;
            gain2Sum += gainDiff*gainDiff;
        }
        averageGainVar = gain2Sum/(gains.size()-1);
        averageGainStdDev = std::sqrt(averageGainVar);
        averageGainErr = averageGainStdDev/std::sqrt(gains.size());

        double gainRelErr = averageGainErr/averageGain;

        //Efficiencies
        collectionEff = calculateEfficiencyStats(numCollected, numInitialElectrons);
        detectionEff = calculateEfficiencyStats(numAboveThreshold, numCollected);        

        //Net efficiency
        netEfficiency.meanValue = collectionEff.meanValue*detectionEff.meanValue;
        netEfficiency.minValue = collectionEff.minValue*detectionEff.minValue;
        netEfficiency.maxValue = collectionEff.maxValue*detectionEff.maxValue;

        netEfficiency.lowError = netEfficiency.meanValue - netEfficiency.minValue;
        netEfficiency.highError = netEfficiency.maxValue - netEfficiency.meanValue;

        collEffErr = std::max(collectionEff.lowError, collectionEff.highError);
        detectEffErr = std::max(detectionEff.lowError, detectionEff.highError);
        netEffErr = std::max(netEfficiency.lowError, netEfficiency.highError);
        

        if(netEffErr <= 0.01 && gainRelErr <= 0.1){
            runAvalanche = false;
        }


    }//End of electron avalanches

    std::cerr << "Number of surpressed Garfield errors: " << numFailure << std::endl;

    //***** Output results to a file *****//
    //create output file
    std::string dataFilename = "effGainResults.dat";
    std::string dataPath = "../../Data/"+dataFilename;
    std::ofstream dataFile;

    //Write results to file
    dataFile.open(dataPath);
    if(!dataFile.is_open()){
        std::cerr << "Error: Could not open file: " << dataPath << std::endl;
        return -1;
    }

    //Electron results

    dataFile << "Simulation information:\n";

    dataFile << "runNumber = " << simParams->runNumber << "\n";
    dataFile << "fieldRatio = " << simParams->fieldRatio << "\n";
    dataFile << "threshold = " << minimumThreshold << "\n";
    dataFile << "numAvalanche = " << simParams->numAvalanche << "\n";

    dataFile << "numInitial = " << numInitialElectrons << "\n";
    dataFile << "numTrials = " << numTotalTrials << "\n";
    dataFile << "numFailure = " << numFailure << "\n";
    dataFile << "numAttached = " << numAttached << "\n";
    dataFile << "numCollected = " << numCollected << "\n";
    dataFile << "numHitGrid = " << numHitGrid << "\n";
    dataFile << "numAboveThreshold = " << numAboveThreshold << "\n";
    dataFile << "numHitLimit = " << numHitLimit << "\n";
    
    dataFile << "averageGain = " << averageGain << "\n";
    dataFile << "averageGainErr = " << averageGainErr << "\n";
    dataFile << "collectionEff = " << collectionEff.meanValue << "\n";
    dataFile << "collectionEffErr = " << collEffErr << "\n";
    dataFile << "detectionEff = " << detectionEff.meanValue << "\n";
    dataFile << "detectionEffErr = " << detectEffErr << "\n";
    dataFile << "netEff = " << netEfficiency.meanValue << "\n";
    dataFile << "netEffErr = " << netEffErr << "\n\n";

    //Raw gain values
    dataFile << "[RAWGAINS]\n";
    for(uint16_t g : gains){
        dataFile << g << "\n";
    }

    dataFile.close();

    return 0;

}