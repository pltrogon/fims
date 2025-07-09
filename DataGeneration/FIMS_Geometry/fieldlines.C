#include <iostream>
#include <fstream>
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
  const double xmin = -x1*.999;
  const double xmax = x1*.999;
  const double ymin = -y1*.999;
  const double ymax = y1*.999;
  const double zmin = z0*.999;
  const double zmax = z1*.999;
  int num = 250;
  const double micronToCM = .0001;
  
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
  drift.SetMaximumStepSize(1*micronToCM);
  std::cout << "step size set to " << 1*micronToCM << "\n";
  std::vector<std::array<float, 3> > Line;
  std::ofstream driftline;
  std::cout << "driftlines initialized\n";
  
  driftline.open("output_file/driftline.csv");
  int prevProg = 0;
  
//loop for finding all of the field points and storing them on a CSV
  for(int count=1; count<=num; ++count)
      {
      drift.FieldLine(0, ymin + 2*count*ymax/num, zmax/2, Line);
      auto size = Line.size();
      //std::cout << "The size of the list is:" << size << "\n";
      
      for(int i = 0; i<size; ++i)
        {
        driftline << Line[i][0] << "," << Line[i][1] << "," << Line[i][2] << "\n";
        }
      int driftprogress = (100*count)/num;
      
      if(driftprogress % 10 == 0 and driftprogress != prevProg){
          std::cout << "Driftline Progress: " << driftprogress << "%" << "\n";
          prevProg = driftprogress;
          }
      }
  driftline.close();
  
}
