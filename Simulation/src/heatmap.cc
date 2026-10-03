// Garfield includes
#include "Garfield/ComponentElmer.hh"
#include "Garfield/AvalancheMicroscopic.hh"
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/Sensor.hh"
#include "Garfield/ViewDrift.hh"

// Custom functions header
#include "myFunctions.hh"

// ROOT includes
#include "TFile.h"
#include "TTree.h"
#include "TString.h"

// C++ Standard Libraries
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <cstdlib>
#include <array>

using namespace Garfield;

int main(int argc, char* argv[]) {

  if (argc != 2) {
    std::cerr << "Format: " << argv[0] << " <GeometryMode>" << std::endl;
    return -1;
  }

  std::string geometryModeString = argv[1];
  GeometryMode geometryMode = stringToGeometryMode(argv[1]);
  if (geometryMode == GeometryMode::Unknown) {
    std::cerr << "Error: Invalid geometryMode: " << argv[1] << std::endl;
    return -1;
  }

  double cellXScale = (geometryMode == GeometryMode::Square || 
                       geometryMode == GeometryMode::SquareSurrounding) ? 0.5 : (1. / sqrt(3.));

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  // Read simulation parameters from config
  auto simParams = readSimulationParameters();
  if (!simParams) return -1;

  int runNo = simParams->runNumber;
  TString gitVersion = getGitVersion().c_str();

  std::cout << "Starting electron drift simulation run: " << runNo << "\n";

  // Output ROOT file
  std::string dataPath = "../../Data/collectionHeatmap.root";
  TFile* dataFile = new TFile(dataPath.c_str(), "RECREATE");

  // --- Trees Setup ---
  // 1. Electron Endpoints Tree
  int electronID;
  double xi, yi, zi, ti, Ei;
  double xf, yf, zf, tf, Ef;
  int stat;

  TTree* electronDataTree = new TTree("electronDataTree", "Primary Electron Endpoints");
  electronDataTree->Branch("Electron ID", &electronID, "electronID/I");
  electronDataTree->Branch("Initial x", &xi, "xi/D");
  electronDataTree->Branch("Initial y", &yi, "yi/D");
  electronDataTree->Branch("Initial z", &zi, "zi/D");
  electronDataTree->Branch("Initial Time", &ti, "ti/D");
  electronDataTree->Branch("Initial Energy", &Ei, "Ei/D");
  electronDataTree->Branch("Final x", &xf, "xf/D");
  electronDataTree->Branch("Final y", &yf, "yf/D");
  electronDataTree->Branch("Final z", &zf, "zf/D");
  electronDataTree->Branch("Final Time", &tf, "tf/D");
  electronDataTree->Branch("Final Energy", &Ef, "Ef/D");
  electronDataTree->Branch("Exit Status", &stat, "stat/I");

  // 2. Electron Trajectory Points Tree
  float driftX, driftY, driftZ;
  TTree* electronTrackTree = new TTree("electronTrackTree", "Primary Electron Trajectories");
  electronTrackTree->Branch("Electron ID", &electronID, "electronID/I");
  electronTrackTree->Branch("Drift x", &driftX, "driftX/F");
  electronTrackTree->Branch("Drift y", &driftY, "driftY/F");
  electronTrackTree->Branch("Drift z", &driftZ, "driftZ/F");

  // Initialize Gas
  MediumMagboltz* gasFIMS = initializeGas(*simParams);

  // Calculate gas transport/diffusion parameters for drift and amplification fields
  double vx, vy, wv, wr;
  double alpha, eta, riontof, ratttof, lor;
  double vxerr, vyerr, vzerr, wverr, wrerr, dlerr, dterr;
  double alphaerr, etaerr, riontoferr, ratttoferr, lorerr, alphatof;
  std::array<double, 6> difftens;

  double driftDiffusionL = 0., driftDiffusionT = 0., driftVelocity = 0.;
  gasFIMS->RunMagboltz(
    simParams->driftField, 0., 0., 1, true,
    vx, vy, driftVelocity, wv, wr, 
    driftDiffusionL, driftDiffusionT,
    alpha, eta, riontof, ratttof, lor, 
    vxerr, vyerr, vzerr, wverr, wrerr, dlerr, dterr,
    alphaerr, etaerr, riontoferr, ratttoferr, lorerr, alphatof,
    difftens
  );

  double ampDiffusionL = 0., ampDiffusionT = 0., ampVelocity = 0.;
  double ampField = simParams->driftField * simParams->fieldRatio;
  gasFIMS->RunMagboltz(
    ampField, 0., 0., 1, true,
    vx, vy, ampVelocity, wv, wr, 
    ampDiffusionL, ampDiffusionT,
    alpha, eta, riontof, ratttof, lor, 
    vxerr, vyerr, vzerr, wverr, wrerr, dlerr, dterr,
    alphaerr, etaerr, riontoferr, ratttoferr, lorerr, alphatof,
    difftens
  );

  // 3. Metadata Tree
  TTree* metaDataTree = new TTree("metaDataTree", "Simulation Parameters");
  metaDataTree->Branch("Git Version", &gitVersion);
  metaDataTree->Branch("runNo", &runNo, "runNo/I");

  metaDataTree->Branch("Pad Length", &simParams->padLength, "padLength/D");
  metaDataTree->Branch("Pitch", &simParams->pitch, "pitch/D");
  metaDataTree->Branch("Amplification Gap", &simParams->amplificationGap, "amplificationGap/D");
  metaDataTree->Branch("Grid Thickness", &simParams->gridThickness, "gridThickness/D");
  metaDataTree->Branch("Pad Thickness", &simParams->padThickness, "padThickness/D");
  metaDataTree->Branch("Hole Radius", &simParams->holeRadius, "holeRadius/D");
  metaDataTree->Branch("Drift Length", &simParams->driftLength, "driftLength/D");
  metaDataTree->Branch("Thickness SiO2", &simParams->thicknessSiO2, "thicknessSiO2/D");
  metaDataTree->Branch("Pillar Radius", &simParams->pillarRadius, "pillarRadius/D");

  metaDataTree->Branch("Electric Field Ratio", &simParams->fieldRatio, "fieldRatio/D");
  metaDataTree->Branch("Drift Field", &simParams->driftField, "driftField/D");
  metaDataTree->Branch("Amplification Field", &ampField, "ampField/D");

  metaDataTree->Branch("Number of Field Lines", &simParams->numFieldLine, "numFieldLine/I");
  metaDataTree->Branch("Number of Avalanches", &simParams->numAvalanche, "numAvalanche/I");
  metaDataTree->Branch("Avalanche Limit", &simParams->avalancheLimit, "avalancheLimit/I");
  metaDataTree->Branch("Initial Z Fraction", &simParams->initialZFraction, "initialZFraction/D");
  
  metaDataTree->Branch("Gas Comp: Ar", &simParams->gasCompAr, "gasCompAr/D");
  metaDataTree->Branch("Gas Comp: CO2", &simParams->gasCompCO2, "gasCompCO2/D");
  metaDataTree->Branch("Gas Comp: CF4", &simParams->gasCompCF4, "gasCompCF4/D");
  metaDataTree->Branch("Gas Comp: Isobutane", &simParams->gasCompIsobutane, "gasCompIsobutane/D");
  metaDataTree->Branch("Gas Penning", &simParams->gasPenning, "gasPenning/D");

  metaDataTree->Branch("Drift Velocity (Drift)", &driftVelocity, "driftVelocity/D");
  metaDataTree->Branch("Diffusion L (Drift)", &driftDiffusionL, "driftDiffusionL/D");
  metaDataTree->Branch("Diffusion T (Drift)", &driftDiffusionT, "driftDiffusionT/D");
  
  metaDataTree->Branch("Drift Velocity (Amplify)", &ampVelocity, "ampVelocity/D");
  metaDataTree->Branch("Diffusion L (Amplify)", &ampDiffusionL, "ampDiffusionL/D");
  metaDataTree->Branch("Diffusion T (Amplify)", &ampDiffusionT, "ampDiffusionT/D");

  metaDataTree->Fill();

  // Import Elmer Field Map
  std::string geometryPath = "../Geometry/";
  std::string elmerResultsPath = geometryPath + "elmerResults/";
  std::string fieldPath = elmerResultsPath + geometryModeString + ".result";

  ComponentElmer fieldFIMS(
    elmerResultsPath + "mesh.header",
    elmerResultsPath + "mesh.elements",
    elmerResultsPath + "mesh.nodes", 
    geometryPath + "dielectrics.dat",
    fieldPath, 
    "mum"
  );

  fieldFIMS.EnableMirrorPeriodicityX();
  fieldFIMS.EnableMirrorPeriodicityY();
  fieldFIMS.SetGas(gasFIMS);

  // Setup Sensor Boundary
  double xmin, ymin, zmin, xmax, ymax, zmax;
  fieldFIMS.GetBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax);

  Sensor sensorFIMS;
  sensorFIMS.AddComponent(&fieldFIMS);
  sensorFIMS.SetArea(
    -2. * simParams->pitch, -2. * simParams->pitch, zmin,
     2. * simParams->pitch,  2. * simParams->pitch, zmax
  );

  // Microscopic Electron Transport Setup
  AvalancheMicroscopic aval;
  aval.SetSensor(&sensorFIMS);
  aval.EnableAvalanche(false); // Disables secondary ionization/gain

  ViewDrift viewDrift;
  aval.EnablePlotting(&viewDrift);

  // Drift Parameters
  double z0 = simParams->initialZFraction * simParams->driftLength;
  double t0 = 0.0;
  double e0 = 0.1; // Initial energy [eV]
  double dx0 = 0.0, dy0 = 0.0, dz0 = 0.0;
  bool distOnPlane = true;
  double cellLength = simParams->pitch * cellXScale;

  int numElectrons = simParams->numAvalanche; // Total primary electrons to simulate
  std::cout << "Drifting " << numElectrons << " single primary electrons..." << std::endl;

  for (int i = 0; i < numElectrons; ++i) {
    electronID = i;

    // Sample initial XY coordinate over unit cell
    auto [x0, y0] = distOnPlane 
      ? randomXYinGeometry(geometryMode, cellLength)
      : std::pair{0.0, 0.0};

    // Drift single primary electron
    aval.DriftElectron(x0, y0, z0, t0, e0, dx0, dy0, dz0);

    // Record Endpoint
    if (aval.GetNumberOfElectronEndpoints() > 0) {
      aval.GetElectronEndpoint(0, xi, yi, zi, ti, Ei, xf, yf, zf, tf, Ef, stat);
      electronDataTree->Fill();
    }

    // Record full trajectory path points
    int nDriftLines = viewDrift.GetNumberOfDriftLines();
    for (int iLine = 0; iLine < nDriftLines; ++iLine) {
      bool isElectron;
      std::vector<std::array<float, 3>> driftPts;
      viewDrift.GetDriftLine(iLine, driftPts, isElectron);

      for (const auto& pt : driftPts) {
        driftX = pt[0];
        driftY = pt[1];
        driftZ = pt[2];
        electronTrackTree->Fill();
      }
    }

    viewDrift.Clear(); // Clear recorded points for next drift iteration

    if ((i + 1) % (numElectrons / 10 + 1) == 0) {
      std::cout << "Progress: " << (100 * (i + 1)) / numElectrons << " %" << std::endl;
    }
  }

  // Save Trees to File
  dataFile->cd();
  metaDataTree->Write();
  electronDataTree->Write();
  electronTrackTree->Write();

  dataFile->Close();
  delete dataFile;
  delete gasFIMS;

  return 0;
}
