#include <iostream>
#include <cmath>
#include <TCanvas.h>
#include <TApplication.h>
#include <TFile.h>
#include "Garfield/MediumMagboltz.hh"
#include "Garfield/ComponentElmer.hh"
#include "Garfield/ComponentElmer2d.hh"
#include "Garfield/ViewField.hh"
#include "Garfield/Plotting.hh"
#include "Garfield/ViewFEMesh.hh"
#include "Garfield/GarfieldConstants.hh"

#include "Garfield/Sensor.hh"
#include "Garfield/DriftLineRKF.hh"

using namespace Garfield;

int main(int argc, char * argv[]) {

  TApplication app("app", &argc, argv);
  std::cout << "----------------------------------------------------------------\n"; 
  // Setup the gas.
  MediumMagboltz gas("he", 70., "co2", 30.);
  gas.SetTemperature(293.15);
  gas.SetPressure(760.);
  gas.SetMaxElectronEnergy(400);
  gas.Initialise(true);
  
    // Plot the field lines
  constexpr bool plotField = true;
std::cout << "----------------------------------------------------------------\n";  
    // Import an Elmer-created field map.
  ComponentElmer* elm = new ComponentElmer("input_file/mesh.header", "input_file/mesh.elements", "input_file/mesh.nodes", "input_file/dielectrics.dat", "input_file/FIMS.result", "mum");
    
  // Set relevant mesh parameters [cm] and the number of field lines
  double x0, y0 ,z0 ,x1 ,y1 ,z1;
  elm->GetBoundingBox(x0, y0 ,z0 ,x1 ,y1 ,z1);
  //constexpr double pitch = .001249;
  double pitch  = double(x1)-.0001;
  const double xmin = -1*pitch;
  const double xmax =  1*pitch;
  const double ymin = -1*pitch;
  const double ymax =  1*pitch;
  const double zmin = double(z0)+.00001;
  const double zmax =  double(z1)-.00001;
  int num = 999;
  
  std::cout << "----------------------------------------------------------------\n";
  // Get the extent of the field map
  elm->SetMedium(0,&gas);
  elm->EnableMirrorPeriodicityX();
  elm->EnableMirrorPeriodicityY();
  elm->PrintRange();
  std::cout << "----------------------------------------------------------------\n";
  elm->PrintMaterials();
  std::cout << "----------------------------------------------------------------\n";
 //setup the sensor
 Sensor sensor;
  sensor.AddComponent(elm);
  sensor.SetArea(xmin, ymin, zmin, xmax, ymax, zmax);
  std::cout << "Sensor Setup\n";
//create a vector of arrays that can store the data points along each field line
  DriftLineRKF drift(&sensor);
  std::cout << "RKF initialized\n";
  drift.SetMaximumStepSize((zmax-zmin)/1000);
  std::cout << "step size set\n";
  std::vector<std::array<float, 3> > Line;
  std::cout << "vector created\n";
  std::ofstream driftline;
  std::cout << "driftlines made\n";
  driftline.open("output_file/driftline.csv");
//loop for finding all of the field points and storing them on a CSV
  for(int count=0; count<=num; ++count)
      {
      drift.FieldLine(xmin+2*count*xmax/num, 0, zmax, Line);
      auto size = Line.size();
      std::cout << "The size of the list is:" << size << "\n";
      std::cout << "The points of the field line are: \n";
      for(int i = 0; i<size; ++i)
        {
        std::cout << "x = " << Line[i][0] << ",  z = " << Line[i][2] << "\n";
        driftline << Line[i][0] << "," << Line[i][2] << "\n";
        }
      std::cout << "----------------------------------------------------------------\n";
      }
   driftline.close();
  
}
