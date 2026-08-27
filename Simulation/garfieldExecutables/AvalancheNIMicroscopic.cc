#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <fstream>
#include <string>

#include "AvalancheNIMicroscopic.hh"
#include "Garfield/FundamentalConstants.hh"
#include "Random.hh"

namespace {

void PrintStatus(const std::string& hdr, const std::string& status,
                 const double x, const double y, const double z,
                 const bool hole) {
  const std::string eh = hole ? "Hole " : "Electron ";
  std::cout << hdr << eh << status << " at " << x << ", " << y << ", " << z
            << "\n";
}

double Mag(const std::array<double, 3>& x) {
  return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

void AddSignalPoint(Garfield::Sensor* sensor, const double q, const double t0,
                    const double dt, const double x0, const double y0,
                    const double z0, const double x1, const double y1,
                    const double z1) {
  if (!sensor || dt <= 0.) return;
  const std::vector<double> ts = {t0, t0 + dt};
  const std::vector<std::array<double, 3> > xs = {{x0, y0, z0}, {x1, y1, z1}};
  sensor->AddSignalWeightingPotential(q, ts, xs);
}

void AddLegacyMarker(Garfield::ViewDrift* view, const char type,
                    const double x, const double y, const double z) {
  if (!view) return;
  switch (type) {
    case 'i':
      view->AddIonisation(x, y, z);
      break;
    case 'a':
      view->AddAttachment(x, y, z);
      break;
    case 'e':
      view->AddExcitation(x, y, z);
      break;
    default:
      break;
  }
}
}

namespace Garfield {

AvalancheNIMicroscopic::AvalancheNIMicroscopic() {
  m_endpointsElectrons.reserve(10000);
  m_endpointsHoles.reserve(10000);
  m_photons.reserve(1000);
}

void AvalancheNIMicroscopic::SetSensor(Sensor* s) {
  if (!s) {
    std::cerr << m_className << "::SetSensor: Null pointer.\n";
    return;
  }
  m_sensor = s;
}

void AvalancheNIMicroscopic::SetNegativeIonMass(int Z){
  double niMass = double(Z)/2.*ProtonMass+double(Z)/2.*NeutronMass+ElectronMass; // eV
  negativeionMass = niMass;
}

void AvalancheNIMicroscopic::EnablePlotting(ViewDrift* view,
                                           const std::size_t nColl) {
  if (!view) {
    std::cerr << m_className << "::EnablePlotting: Null pointer.\n";
    return;
  }

  m_viewer = view;
  m_usePlotting = true;
  m_nCollSkip = static_cast<unsigned int>(nColl);
  if (!m_useDriftLines) {
    std::cout << m_className << "::EnablePlotting:\n"
              << "    Enabling storage of drift line.\n";
    EnableDriftLines(true);
  }
}

void AvalancheNIMicroscopic::DisablePlotting() {
  m_viewer = nullptr;
  m_usePlotting = false;
}

void AvalancheNIMicroscopic::EnableElectronEnergyHistogramming(TH1* histo) {
  if (!histo) {
    std::cerr << m_className << "::EnableElectronEnergyHistogramming:\n"
              << "    Null pointer.\n";
    return;
  }

  m_histElectronEnergy = histo;
}

void AvalancheNIMicroscopic::EnableHoleEnergyHistogramming(TH1* histo) {
  if (!histo) {
    std::cerr << m_className << "::EnableHoleEnergyHistogramming:\n"
              << "    Null pointer.\n";
    return;
  }

  m_histHoleEnergy = histo;
}

void AvalancheNIMicroscopic::SetDistanceHistogram(TH1* histo, const char opt) {
  if (!histo) {
    std::cerr << m_className << "::SetDistanceHistogram: Null pointer.\n";
    return;
  }

  m_histDistance = histo;

  if (opt == 'x' || opt == 'y' || opt == 'z' || opt == 'r') {
    m_distanceOption = opt;
  } else {
    std::cerr << m_className << "::SetDistanceHistogram:";
    std::cerr << "    Unknown option " << opt << ".\n";
    std::cerr << "    Valid options are x, y, z, r.\n";
    std::cerr << "    Using default value (r).\n";
    m_distanceOption = 'r';
  }

  if (m_distanceHistogramType.empty()) {
    std::cout << m_className << "::SetDistanceHistogram:\n";
    std::cout << "    Don't forget to call EnableDistanceHistogramming.\n";
  }
}

void AvalancheNIMicroscopic::EnableDistanceHistogramming(const int type) {
  // Check if this type of collision is already registered
  // for histogramming.
  const unsigned int nDistanceHistogramTypes = m_distanceHistogramType.size();
  if (nDistanceHistogramTypes > 0) {
    for (unsigned int i = 0; i < nDistanceHistogramTypes; ++i) {
      if (m_distanceHistogramType[i] != type) continue;
      std::cout << m_className << "::EnableDistanceHistogramming:\n";
      std::cout << "    Collision type " << type
                << " is already being histogrammed.\n";
      return;
    }
  }

  m_distanceHistogramType.push_back(type);
  std::cout << m_className << "::EnableDistanceHistogramming:\n";
  std::cout << "    Histogramming of collision type " << type << " enabled.\n";
  if (!m_histDistance) {
    std::cout << "    Don't forget to set the histogram.\n";
  }
}

void AvalancheNIMicroscopic::DisableDistanceHistogramming(const int type) {
  if (std::find(m_distanceHistogramType.begin(), m_distanceHistogramType.end(),
                type) == m_distanceHistogramType.end()) {
    std::cerr << m_className << "::DisableDistanceHistogramming:\n"
              << "    Collision type " << type << " is not histogrammed.\n";
    return;
  }

  m_distanceHistogramType.erase(
      std::remove(m_distanceHistogramType.begin(),
                  m_distanceHistogramType.end(), type),
      m_distanceHistogramType.end());
}

void AvalancheNIMicroscopic::DisableDistanceHistogramming() {
  m_histDistance = nullptr;
  m_distanceHistogramType.clear();
}

void AvalancheNIMicroscopic::EnableSecondaryEnergyHistogramming(TH1* histo) {
  if (!histo) {
    std::cerr << m_className << "::EnableSecondaryEnergyHistogramming:\n"
              << "    Null pointer.\n";
    return;
  }

  m_histSecondary = histo;
}

void AvalancheNIMicroscopic::SetTimeWindow(const double t0, const double t1) {
  if (fabs(t1 - t0) < Small) {
    std::cerr << m_className << "::SetTimeWindow:\n";
    std::cerr << "    Time interval must be greater than zero.\n";
    return;
  }

  m_tMin = std::min(t0, t1);
  m_tMax = std::max(t0, t1);
  m_hasTimeWindow = true;
}

void AvalancheNIMicroscopic::SetTimeStep(const double d) {
  if (d < Small) {
    std::cerr << m_className << "::SetTimeStep:\n    "
              << "Step size is too small. Using default (10 ns) instead.\n";
    return;
  }
  if (m_debug) {
    std::cout << m_className << "::SetTimeStep:\n"
              << "    Step size set to " << d << " ns.\n";
  }
  m_tMicroscopic = d;
}

void AvalancheNIMicroscopic::SetDistanceSteps(const double d) {
  if (d < Small) {
    std::cerr << m_className << "::SetDistanceSteps:\n    "
              << "Step size is too small. Using default (10 um) instead.\n";
    m_ionStep = 0.001;
    return;
  }
  if (m_debug) {
    std::cout << m_className << "::SetDistanceSteps:\n"
              << "    Step size set to " << d << " cm.\n";
  }
  m_ionStep = d;
}


void AvalancheNIMicroscopic::UnsetTimeWindow() { m_hasTimeWindow = false; }

void AvalancheNIMicroscopic::GetElectronEndpoint(const unsigned int i, double& x0,
                                               double& y0, double& z0,
                                               double& t0, double& e0,
                                               double& x1, double& y1,
                                               double& z1, double& t1,
                                               double& e1, int& status) const {
  if (i >= m_endpointsElectrons.size()) {
    std::cerr << m_className << "::GetElectronEndpoint: Index out of range.\n";
    x0 = y0 = z0 = t0 = e0 = 0.;
    x1 = y1 = z1 = t1 = e1 = 0.;
    status = 0;
    return;
  }

  x0 = m_endpointsElectrons[i].x0;
  y0 = m_endpointsElectrons[i].y0;
  z0 = m_endpointsElectrons[i].z0;
  t0 = m_endpointsElectrons[i].t0;
  e0 = m_endpointsElectrons[i].e0;
  x1 = m_endpointsElectrons[i].x;
  y1 = m_endpointsElectrons[i].y;
  z1 = m_endpointsElectrons[i].z;
  t1 = m_endpointsElectrons[i].t;
  e1 = m_endpointsElectrons[i].energy;
  status = m_endpointsElectrons[i].status;
}

void AvalancheNIMicroscopic::GetElectronEndpoint(
    const unsigned int i, double& x0, double& y0, double& z0, double& t0,
    double& e0, double& x1, double& y1, double& z1, double& t1, double& e1,
    double& dx1, double& dy1, double& dz1, int& status) const {
  if (i >= m_endpointsElectrons.size()) {
    std::cerr << m_className << "::GetElectronEndpoint: Index out of range.\n";
    x0 = y0 = z0 = t0 = e0 = 0.;
    x1 = y1 = z1 = t1 = e1 = 0.;
    dx1 = dy1 = dz1 = 0.;
    status = 0;
    return;
  }

  x0 = m_endpointsElectrons[i].x0;
  y0 = m_endpointsElectrons[i].y0;
  z0 = m_endpointsElectrons[i].z0;
  t0 = m_endpointsElectrons[i].t0;
  e0 = m_endpointsElectrons[i].e0;
  x1 = m_endpointsElectrons[i].x;
  y1 = m_endpointsElectrons[i].y;
  z1 = m_endpointsElectrons[i].z;
  t1 = m_endpointsElectrons[i].t;
  e1 = m_endpointsElectrons[i].energy;
  dx1 = m_endpointsElectrons[i].kx;
  dy1 = m_endpointsElectrons[i].ky;
  dz1 = m_endpointsElectrons[i].kz;
  status = m_endpointsElectrons[i].status;
}

void AvalancheNIMicroscopic::GetHoleEndpoint(const unsigned int i, double& x0,
                                           double& y0, double& z0, double& t0,
                                           double& e0, double& x1, double& y1,
                                           double& z1, double& t1, double& e1,
                                           int& status) const {
  if (i >= m_endpointsHoles.size()) {
    std::cerr << m_className << "::GetHoleEndpoint: Index out of range.\n";
    x0 = y0 = z0 = t0 = e0 = 0.;
    x1 = y1 = z1 = t1 = e1 = 0.;
    status = 0;
    return;
  }

  x0 = m_endpointsHoles[i].x0;
  y0 = m_endpointsHoles[i].y0;
  z0 = m_endpointsHoles[i].z0;
  t0 = m_endpointsHoles[i].t0;
  e0 = m_endpointsHoles[i].e0;
  x1 = m_endpointsHoles[i].x;
  y1 = m_endpointsHoles[i].y;
  z1 = m_endpointsHoles[i].z;
  t1 = m_endpointsHoles[i].t;
  e1 = m_endpointsHoles[i].energy;
  status = m_endpointsHoles[i].status;
}

unsigned int AvalancheNIMicroscopic::GetNumberOfElectronDriftLinePoints(
    const unsigned int i) const {
  if (i >= m_endpointsElectrons.size()) {
    std::cerr << m_className << "::GetNumberOfElectronDriftLinePoints:\n";
    std::cerr << "    Endpoint index (" << i << ") out of range.\n";
    return 0;
  }

  if (!m_useDriftLines) return 2;

  return m_endpointsElectrons[i].driftLine.size() + 2;
}

unsigned int AvalancheNIMicroscopic::GetNumberOfHoleDriftLinePoints(
    const unsigned int i) const {
  if (i >= m_endpointsHoles.size()) {
    std::cerr << m_className << "::GetNumberOfHoleDriftLinePoints:\n";
    std::cerr << "    Endpoint index (" << i << ") out of range.\n";
    return 0;
  }

  if (!m_useDriftLines) return 2;

  return m_endpointsHoles[i].driftLine.size() + 2;
}

void AvalancheNIMicroscopic::GetElectronDriftLinePoint(
    double& x, double& y, double& z, double& t, int& changePoint, const int ip,
    const unsigned int iel) const {
  if (iel >= m_endpointsElectrons.size()) {
    std::cerr << m_className << "::GetElectronDriftLinePoint:\n";
    std::cerr << "    Endpoint index (" << iel << ") out of range.\n";
    return;
  }

  if (ip <= 0) {
    x = m_endpointsElectrons[iel].x0;
    y = m_endpointsElectrons[iel].y0;
    z = m_endpointsElectrons[iel].z0;
    t = m_endpointsElectrons[iel].t0;
    changePoint = 0;
    return;
  }

  const int np = m_endpointsElectrons[iel].driftLine.size();
  if (ip > np) {
    x = m_endpointsElectrons[iel].x;
    y = m_endpointsElectrons[iel].y;
    z = m_endpointsElectrons[iel].z;
    t = m_endpointsElectrons[iel].t;
    changePoint = 0;
    return;
  }

  x = m_endpointsElectrons[iel].driftLine[ip - 1].x;
  y = m_endpointsElectrons[iel].driftLine[ip - 1].y;
  z = m_endpointsElectrons[iel].driftLine[ip - 1].z;
  t = m_endpointsElectrons[iel].driftLine[ip - 1].t;
  changePoint = m_endpointsElectrons[iel].driftLine[ip - 1].changePoint;
}

void AvalancheNIMicroscopic::GetHoleDriftLinePoint(double& x, double& y,
                                                 double& z, double& t,
                                                 const int ip,
                                                 const unsigned int ih) const {
  if (ih >= m_endpointsHoles.size()) {
    std::cerr << m_className << "::GetHoleDriftLinePoint:\n";
    std::cerr << "    Endpoint index (" << ih << ") out of range.\n";
    return;
  }

  if (ip <= 0) {
    x = m_endpointsHoles[ih].x0;
    y = m_endpointsHoles[ih].y0;
    z = m_endpointsHoles[ih].z0;
    t = m_endpointsHoles[ih].t0;
    return;
  }

  const int np = m_endpointsHoles[ih].driftLine.size();
  if (ip > np) {
    x = m_endpointsHoles[ih].x;
    y = m_endpointsHoles[ih].y;
    z = m_endpointsHoles[ih].z;
    t = m_endpointsHoles[ih].t;
    return;
  }

  x = m_endpointsHoles[ih].driftLine[ip - 1].x;
  y = m_endpointsHoles[ih].driftLine[ip - 1].y;
  z = m_endpointsHoles[ih].driftLine[ip - 1].z;
  t = m_endpointsHoles[ih].driftLine[ip - 1].t;
}

void AvalancheNIMicroscopic::GetPhoton(const unsigned int i, double& e,
                                     double& x0, double& y0, double& z0,
                                     double& t0, double& x1, double& y1,
                                     double& z1, double& t1,
                                     int& status) const {
  if (i >= m_photons.size()) {
    std::cerr << m_className << "::GetPhoton: Index out of range.\n";
    return;
  }

  x0 = m_photons[i].x0;
  x1 = m_photons[i].x1;
  y0 = m_photons[i].y0;
  y1 = m_photons[i].y1;
  z0 = m_photons[i].z0;
  z1 = m_photons[i].z1;
  t0 = m_photons[i].t0;
  t1 = m_photons[i].t1;
  status = m_photons[i].status;
  e = m_photons[i].energy;
}

void AvalancheNIMicroscopic::SetUserHandleStep(
    void (*f)(double x, double y, double z, double t, double e, double dx,
              double dy, double dz, bool hole)) {
  if (!f) {
    std::cerr << m_className << "::SetUserHandleStep: Null pointer.\n";
    return;
  }
  m_userHandleStep = f;
}

void AvalancheNIMicroscopic::SetUserHandleCollision(void (*f)(
    double x, double y, double z, double t, int type, int level, Medium* m,
    double e0, double e1, double dx0, double dy0, double dz0, 
    double dx1, double dy1, double dz1)) {
  m_userHandleCollision = f;
}

void AvalancheNIMicroscopic::SetUserHandleAttachment(void (*f)(
    double x, double y, double z, double t, int type, int level, Medium* m)) {
  m_userHandleAttachment = f;
}

void AvalancheNIMicroscopic::SetUserHandleInelastic(void (*f)(
    double x, double y, double z, double t, int type, int level, Medium* m)) {
  m_userHandleInelastic = f;
}

void AvalancheNIMicroscopic::SetUserHandleIonisation(void (*f)(
    double x, double y, double z, double t, int type, int level, Medium* m)) {
  m_userHandleIonisation = f;
}

bool AvalancheNIMicroscopic::DriftElectron(const double x0, const double y0,
                                         const double z0, const double t0,
                                         const double e0, const double dx0,
                                         const double dy0, const double dz0) {
  // Clear the list of electrons and photons.
  m_endpointsElectrons.clear();
  m_endpointsHoles.clear();
  m_photons.clear();

  // Reset the particle counters.
  m_nElectrons = m_nHoles = m_nIons = 0;

  return TransportNegativeIon(x0, y0, z0, t0, e0, dx0, dy0, dz0, false, false);
}

bool AvalancheNIMicroscopic::AvalancheNegativeIon(const double x0, const double y0,
                                             const double z0, const double t0,
                                             const double e0, const double dx0,
                                             const double dy0,
                                             const double dz0) {
  // Clear the list of electrons, holes and photons.
  m_endpointsElectrons.clear();
  m_endpointsHoles.clear();
  m_photons.clear();

  // Reset the particle counters.
  m_nElectrons = m_nHoles = m_nIons = 0;

  return TransportNegativeIon(x0, y0, z0, t0, e0, dx0, dy0, dz0, true, false);
}

bool AvalancheNIMicroscopic::TransportNegativeIon(const double x0, const double y0,
                                             const double z0, const double t0,
                                             double e0, const double dx0,
                                             const double dy0, const double dz0,
                                             const bool aval, bool hole0) {
  const std::string hdr = m_className + "::TransportNegativeIon: ";
  // Make sure that the sensor is defined.
  if (!m_sensor) {
    std::cerr << hdr << "Sensor is not defined.\n";
    return false;
  }

  // Make sure that the starting point is inside a medium.
  Medium* medium = m_sensor->GetMedium(x0, y0, z0);
  if (!medium) {
    std::cerr << hdr << "No medium at initial position.\n";
    return false;
  }

  // Make sure that the medium is "driftable" and microscopic.
  if (!medium->IsDriftable() || !medium->IsMicroscopic()) {
    std::cerr << hdr << "Medium does not have cross-section data.\n";
    return false;
  }

  // If the medium is a semiconductor, we may use "band structure" stepping.
  bool useBandStructure =
      medium->IsSemiconductor() && m_useBandStructureDefault;
  if (m_debug) {
    std::cout << hdr << "Start drifting in medium " << medium->GetName()
              << ".\n";
  }

  // Get the id number of the drift medium.
  int id = medium->GetId();

  // Numerical prefactors in equation of motion
  const double c1 = SpeedOfLight * sqrt(2. / ElectronMass);
  const double c2 = c1 * c1 / 4.;

  // Electric and magnetic field
  double ex = 0., ey = 0., ez = 0.;
  double bx = 0., by = 0., bz = 0., bmag = 0.;
  int status = 0;
  // Cyclotron frequency
  double cwt = 1., swt = 0.;
  double wb = 0.;
  // Flag indicating if magnetic field is usable
  bool bOk = true;

  // Direction, velocity and energy after a step
  double newKx = 0., newKy = 0., newKz = 0.;
  double newVx = 0., newVy = 0., newVz = 0.;
  double newEnergy = 0.;

  // Numerical factors
  double a1 = 0., a2 = 0., a3 = 0., a4 = 0.;

  // Make sure the initial energy is positive.
  e0 = std::max(e0, Small);
  
  // Particle Type
  int particle0 = 0; // First Type is Thremal Electron

  std::vector<Electron> stackOld;
  std::vector<Electron> stackNew;
  stackOld.reserve(10000);
  stackNew.reserve(1000);
  std::vector<std::pair<double, double> > stackPhotons;
  std::vector<Medium::Secondary> secondaries;

  // Put the initial electron on the stack.
  if (useBandStructure) {
    // With band structure, (kx, ky, kz) represents the momentum.
    // No normalization in this case.
    int band = -1;
    double kx = 0., ky = 0., kz = 0.;
    medium->GetElectronMomentum(e0, kx, ky, kz, band);
    //AddToStack(x0, y0, z0, t0, e0, kx, ky, kz, band, hole0, stackOld);
    AddToStack(x0, y0, z0, t0, e0, particle0, kx, ky, kz, band, hole0, stackOld);
  } else {
    double kx = dx0;
    double ky = dy0;
    double kz = dz0;
    // Check the given initial direction.
    const double k = sqrt(kx * kx + ky * ky + kz * kz);
    if (fabs(k) < Small) {
      // Direction has zero norm, draw a random direction.
      RndmDirection(kx, ky, kz);
    } else {
      // Normalise the direction to 1.
      kx /= k;
      ky /= k;
      kz /= k;
    }
    //AddToStack(x0, y0, z0, t0, e0, kx, ky, kz, 0, hole0, stackOld);
    AddToStack(x0, y0, z0, t0, e0, particle0, kx, ky, kz, 0, hole0, stackOld);
  }
  if (hole0) {
    ++m_nHoles;
  } else {
    ++m_nElectrons;
  }

  // Get the null-collision rate.
  double fLim = medium->GetElectronNullCollisionRate(stackOld.front().band);
  if (fLim <= 0.) {
    std::cerr << hdr << "Got null-collision rate <= 0.\n";
    return false;
  }
  double fInv = 1. / fLim;

/*
  std::cout << "-------------------------------" <<std::endl;
  std::cout << "initial fInv : " << fInv <<std::endl;
  std::cout << "initial fLim : " << fLim <<std::endl;
*/

  while (true) {
    // Remove all inactive items from the stack.
    stackOld.erase(std::remove_if(stackOld.begin(), stackOld.end(), IsInactive),
                   stackOld.end());
    // Add the electrons produced in the last iteration.
    if (aval && m_sizeCut > 0) {
      // If needed, reduce the number of electrons to add.
      if (stackOld.size() > m_sizeCut) {
        stackNew.clear();
      } else if (stackOld.size() + stackNew.size() > m_sizeCut) {
        stackNew.resize(m_sizeCut - stackOld.size());
      }
    }
    stackOld.insert(stackOld.end(), std::make_move_iterator(stackNew.begin()),
                    std::make_move_iterator(stackNew.end()));
    stackNew.clear();
    // If the list of electrons/holes is exhausted, we're done.
    if (stackOld.empty()) break;
    // Loop over all electrons/holes in the avalanche.
    int LoopCnt = 0;
    for (auto it = stackOld.begin(), end = stackOld.end(); it != end; ++it) {
      //std::cerr << "Loop Count : " << LoopCnt << std::endl;
      LoopCnt++;
      // Get an electron/hole from the stack.
      double x = (*it).x;
      double y = (*it).y;
      double z = (*it).z;
      double t = (*it).t;
      double energy = (*it).energy;
      int particle = (*it).particle;
      int band = (*it).band;
      double kx = (*it).kx;
      double ky = (*it).ky;
      double kz = (*it).kz;
      bool hole = (*it).hole;
      //if(particle==2)std::cerr << "energy(?) : " << energy <<std::endl;

      bool ok = true;

      // Count number of collisions between updates.
      unsigned int nCollTemp = 0;

      // Get the local electric field and medium.
      m_sensor->ElectricField(x, y, z, ex, ey, ez, medium, status);
      // Sign change for electrons.
      if (!hole) {
        ex = -ex;
        ey = -ey;
        ez = -ez;
      }
      /*
      std::cout << "-------------------------------" <<std::endl;
      std::cout << "ID       : " << LoopCnt <<std::endl;
      std::cout << "energy   : " << energy <<std::endl;
      std::cout << "x        : " << x <<std::endl;
      std::cout << "y        : " << y <<std::endl;
      std::cout << "z        : " << z <<std::endl;
      std::cout << "particle : " << particle <<std::endl;
      std::cout << "ex       : " << ex <<std::endl;
      std::cout << "ey       : " << ey <<std::endl;
      std::cout << "ez       : " << ez <<std::endl;
      std::cout << "kx       : " << kx <<std::endl;
      std::cout << "ky       : " << ky <<std::endl;
      std::cout << "kz       : " << kz <<std::endl;
      */

      if (m_debug) {
        const std::string eh = hole ? "hole " : "electron ";
        std::cout << hdr << "\n    Drifting " << eh << it - stackOld.begin()
                  << ".\n    Field [V/cm] at (" << x << ", " << y << ", " << z
                  << "): " << ex << ", " << ey << ", " << ez
                  << "\n    Status: " << status << "\n";
        if (medium) std::cout << "    Medium: " << medium->GetName() << "\n";
      }

      if (status != 0) {
        // Electron is not inside a drift medium.
        Update(it, x, y, z, t, energy, kx, ky, kz, band);
        (*it).status = StatusLeftDriftMedium;
        AddToEndPoints(*it, hole);
        if (m_debug) PrintStatus(hdr, "left the drift medium", x, y, z, hole);
        continue;
      }

      // If switched on, get the local magnetic field.
      if (m_useBfield) {
        m_sensor->MagneticField(x, y, z, bx, by, bz, status);
        const double scale = hole ? Tesla2Internal : -Tesla2Internal;
        bx *= scale;
        by *= scale;
        bz *= scale;
        // Make sure that neither E nor B are zero.
        bmag = sqrt(bx * bx + by * by + bz * bz);
        const double emag2 = ex * ex + ey * ey + ez * ez;
        bOk = (bmag > Small && emag2 > Small);
      }

// Negative Ion Drift
//-----------------------------------------------------

      // Negative Ion Flag
      if(particle==2){
        //std::cerr << "NegativeIon Drift Stepping" <<std::endl;
      
        std::array<double, 3> x_vec = {x, y, z};
        std::array<double, 3> e_vec = {0, 0, 0};
        std::array<double, 3> b_vec = {0, 0, 0};

        status = GetField(x_vec, e_vec, b_vec, medium);
        if (status != 0) {
          std::cerr << m_className + "::"<< hdr << ": " <<  " is not in a valid drift region.\n";
        }
        //std:: cout << "----------" << std::endl;
        //std:: cout << "Electric Field : " << Mag(e_vec) << " V/cm" << std::endl;
        std::array<double, 3> v_vec;
        if (!GetVelocity(3, medium, x_vec, e_vec, b_vec, v_vec)) {
          status = StatusCalculationAbandoned;
          std::cerr << m_className + "::DriftLine: Abandoning the calculation.\n";
          continue;
        }

        // Make sure the drift velocity vector has a non-vanishing component.
        const double vmag = Mag(v_vec);
        //std:: cout << "Velocity : " << vmag*1e6 << " cm/ms" << std::endl;
        //std:: cout << "Mobility : " << vmag*1e9/Mag(e_vec) << " cm^2/V/s" << std::endl;
        if (vmag < Small) {
          std::cerr << m_className + "::DriftLine: Too small drift velocity at.\n";
          status = StatusCalculationAbandoned;
          continue;
        }

        // Determine the time step.
        double dt = Small;
        //dt = m_tMicroscopic; // ns
        dt = m_ionStep / vmag;

      // Electric field
        double nieField = sqrt ( ex*ex + ey*ey + ez*ez );
      
      // Calculate the velocity vector.
        double vx = 0., vy = 0., vz = 0.;
        //const double v = SpeedOfLight * sqrt(2.*energy/negativeionMass);
        
        vx = v_vec[0];
        vy = v_vec[1];
        vz = v_vec[2];
        //std::cout << "dt       : " << dt << " ns" <<std::endl;
        //vx = v * kx;
        //vy = v * ky;
        //vz = v * kz;
        /*
        const double ax = 1 * ex / negativeionMass;
        const double ay = 1 * ey / negativeionMass;
        const double az = 1 * ez / negativeionMass;
        std::cout << "dt       : " << dt  <<std::endl;
        std::cout << "v0       : " << v  <<std::endl;
        std::cout << "vx       : " << vx <<std::endl;
        std::cout << "vy       : " << vy <<std::endl;
        std::cout << "vz       : " << vz <<std::endl;
        std::cout << "ax       : " << ax <<std::endl;
        std::cout << "ay       : " << ay <<std::endl;
        std::cout << "az       : " << az <<std::endl;
        */
        //std::cout << "v0       : " << v <<std::endl;

      // Stepping distance
        /*
        double x1 = x + vx*dt + 1./2. * ax * dt * dt;
        double y1 = y + vy*dt + 1./2. * ay * dt * dt;
        double z1 = z + vz*dt + 1./2. * az * dt * dt;
        */
        double x1 = x + vx*dt;
        double y1 = y + vy*dt;
        double z1 = z + vz*dt;
        double t1 = t + dt;
        
        /*
        newVx = ax*dta + vx;
        newVy = ay*dt + vy;
        newVz = az*dt + vz;
        */
        newVx = vx;
        newVy = vy;
        newVz = vz;
        /*
        std::cout << "newVx    : " << newVx <<std::endl;
        std::cout << "newVy    : " << newVy <<std::endl;
        std::cout << "newVz    : " << newVz <<std::endl;
        */

        double newV2 = (newVx*newVx + newVy*newVy + newVz*newVz);
        
        newKx = newVx/sqrt(newV2);
        newKy = newVy/sqrt(newV2);
        newKz = newVz/sqrt(newV2);
        
        /*
        std::cout << "newkx    : " << newKx <<std::endl;
        std::cout << "newky    : " << newKy <<std::endl;
        std::cout << "newkz    : " << newKz <<std::endl;
        */

        newEnergy = 1./(2*SpeedOfLight*SpeedOfLight) * negativeionMass * newV2;

        std::array<double, 3> x_diff  = {x1, y1, z1};
        //std::array<double, 3> v_vec   = {newVx, newVy, newVz};
        
        //if (!AddNIDiffusion(3,medium, sqrt(vmag * dt), x_diff, v_vec, e_vec, b_vec)) {
        if(m_useDiffusion){
          if (!AddNIDiffusion(3, medium, sqrt(vmag * dt), x_diff, v_vec, e_vec, b_vec)) {
            status = StatusCalculationAbandoned;
            std::cerr << m_className + "::DriftLine: Abandoning the calculation.\n";
            break;
          }
        }
        x1 = x_diff[0];
        y1 = x_diff[1];
        z1 = x_diff[2];
        
        double dStep = sqrt((x-x1)*(x-x1)+(y-y1)*(y-y1)+(z-z1)*(z-z1));
        //std::cout << "dx    : " << dStep << " cm" <<std::endl;
        
        m_sensor->ElectricField(x1, y1, z1, ex, ey, ez, medium, status);
        // Check if the electron is still inside a drift medium/the drift area.
        if (status != 0 || !m_sensor->IsInArea(x1, y1, z1)) {
          // Try to terminate the drift line close to the boundary (endpoint
          // outside the drift medium/drift area) using iterative bisection.
          Terminate(x, y, z, t, x1, y1, z1, t1);
          if (m_useSignal) {
            const int q = hole ? 1 : -1;
            AddSignalPoint(m_sensor, q, t, t1 - t, x, y, z, x1, y1, z1);
          }
          Update(it, x1, y1, z1, t1, newEnergy, newKx, newKy, newKz, band);
          if (status != 0) {
            (*it).status = StatusLeftDriftMedium;
            //PrintStatus(hdr, ":Negative Ion left the drift medium", x1, y1, z1, hole);
            if (m_debug)
              PrintStatus(hdr, ":Negative Ion left the drift medium", x1, y1, z1, hole);
          } else {
            (*it).status = StatusLeftDriftArea;
            //PrintStatus(hdr, ":Negative Ion left the drift area", x1, y1, z1, hole);
            if (m_debug)
              PrintStatus(hdr, ":Negative Ion left the drift area", x1, y1, z1, hole);
          }
          AddToEndPoints(*it, hole);
          continue;
        }
        // Check if the Negative Ion has crossed a wire.
        double xc = x, yc = y, zc = z;
        double rc = 0.;
        if (m_sensor->CrossedWire(x, y, z, x1, y1, z1, xc, yc, zc, false, rc)) {
          // If switched on, calculated the induced signal over this step.
          if (m_useSignal) {
            const double dx = xc - x;
            const double dy = yc - y;
            const double dz = zc - z;
            dt = sqrt(dx * dx + dy * dy + dz * dz) /
                 sqrt(vx * vx + vy * vy + vz * vz);
            const int q = hole ? 1 : -1;
            AddSignalPoint(m_sensor, q, t, dt, x, y, z, xc, yc, zc);
          }
          Update(it, xc, yc, zc, t + dt, newEnergy, newKx, newKy, newKz, band);
          (*it).status = StatusLeftDriftMedium;
          AddToEndPoints(*it, hole);
          if (m_debug) PrintStatus(hdr, "hit a wire", x, y, z, hole);
          continue;
        }
        // If switched on, calculate the induced signal.
        if (m_useSignal) {
          const int q = hole ? 1 : -1;
          AddSignalPoint(m_sensor, q, t, dt, x, y, z, x1, y1, z1);
        }

        bool detachmentFlag = false;
        if(m_useDetachment){
          // Judgment Detachment

          // Cross-Section Model
          if(DetachmentModelNo==0){
            // input x-section
            double Torr = medium->GetPressure();
            double dx = dStep;
            double prob=0;
            double this_xsec = GetDetachCrossSection(newEnergy);
            double this_lambda = ComputeLambda(Torr,this_xsec);
            if(this_lambda <= 0) return -1;

            prob = 1. - exp(-dx/this_lambda);
            //std::cerr << "probability : " << prob <<std::endl;
            if(RndmUniform() < prob ){ //40kV/cm
              detachmentFlag = true;
            }
          }

          // Threshold model (SF6- -> SF6 + e-)
          if(DetachmentModelNo==1){
            double threshold = 40000; // V/cm
            double sigma = 10; // %
            double prob = 0.5*(1.+erf((nieField-threshold)/(threshold*sigma/100.)));
            //std::cerr << "prob = " << prob << " , Efield = " << EfieldAbs << " V/cm" <<std::endl;
            if(RndmUniform() < prob ){ //40kV/cm
              detachmentFlag = true;
            }
          }
        }

        // Update the coordinates.
        x = x1;
        y = y1;
        z = z1;
        t = t1;
        // Detached
        if(detachmentFlag){
          double ethermal = Small;
          int ebs = -1;
          medium->GetElectronMomentum(ethermal, newKx, newKy, newKz, ebs);
          (*it).particle = 0; // Create Electron Detachment
          newEnergy = ethermal; // Thremal Electron
          if(m_debug)PrintStatus(hdr, "Detached",x,y,z,hole);
        }

        kx = newKx;
        ky = newKy;
        kz = newKz;

        // Update the stack.
        Update(it, x, y, z, t, newEnergy, kx, ky, kz, band);
        if (m_useDriftLines) {
          point newPoint;
          newPoint.x = x;
          newPoint.y = y;
          newPoint.z = z;
          newPoint.t = t;
        if((*it).particle==0){
          newPoint.changePoint = 1;
        }else{
          newPoint.changePoint = 0;
        }
          (*it).driftLine.push_back(std::move(newPoint));
        }
        continue; // skip electron avalanche
      }

// Electron Avalanche
//-----------------------------------------------------

      // Trace the electron/hole.
      while (1) {
        bool isNullCollision = false;

        // Make sure the electron energy exceeds the transport cut.
        if (energy < m_deltaCut) {
          Update(it, x, y, z, t, energy, kx, ky, kz, band);
          (*it).status = StatusBelowTransportCut;
          AddToEndPoints(*it, hole);
          if (m_debug) {
            std::cout << hdr << "Kinetic energy (" << energy
              << ") below transport cut.\n";
          }
          ok = false;
          break;
        }

        // Fill the energy distribution histogram.
        if (hole && m_histHoleEnergy) {
          m_histHoleEnergy->Fill(energy);
        } else if (!hole && m_histElectronEnergy) {
          m_histElectronEnergy->Fill(energy);
        }

        // Check if the electrons is within the specified time window.
        if (m_hasTimeWindow && (t < m_tMin || t > m_tMax)) {
          Update(it, x, y, z, t, energy, kx, ky, kz, band);
          (*it).status = StatusOutsideTimeWindow;
          AddToEndPoints(*it, hole);
          if (m_debug) PrintStatus(hdr, "left the time window", x, y, z, hole);
          ok = false;
          break;
        }

        if (medium->GetId() != id) {
          // Medium has changed.
          if (!medium->IsMicroscopic()) {
            // Electron/hole has left the microscopic drift medium.
            Update(it, x, y, z, t, energy, kx, ky, kz, band);
            (*it).status = StatusLeftDriftMedium;
            AddToEndPoints(*it, hole);
            ok = false;
            if (m_debug) {
              std::cout << hdr << "\n    Medium at " << x << ", " << y << ", "
                << z << " does not have cross-section data.\n";
            }
            break;
          }
          id = medium->GetId();
          useBandStructure =
            (medium->IsSemiconductor() && m_useBandStructureDefault);
          // Update the null-collision rate.
          fLim = medium->GetElectronNullCollisionRate(band);
          if (fLim <= 0.) {
            std::cerr << hdr << "Got null-collision rate <= 0.\n";
            return false;
          }
          fInv = 1. / fLim;
        }

        double vx = 0., vy = 0., vz = 0.;
        if (m_useBfield && bOk) {
          // Calculate the cyclotron frequency.
          wb = OmegaCyclotronOverB * bmag;
          // Rotate the direction vector into the local coordinate system.
          ComputeRotationMatrix(bx, by, bz, bmag, ex, ey, ez);
          RotateGlobal2Local(kx, ky, kz);
          // Calculate the electric field in the rotated system.
          RotateGlobal2Local(ex, ey, ez);
          // Calculate the velocity vector in the local frame.
          const double v = c1 * sqrt(energy);
          vx = v * kx;
          vy = v * ky;
          vz = v * kz;
          a1 = vx * ex;
          a2 = c2 * ex * ex;
          a3 = ez / bmag - vy;
          a4 = (ez / wb);
        } else if (useBandStructure) {
          energy = medium->GetElectronEnergy(kx, ky, kz, vx, vy, vz, band);
        } else {
          // No band structure, no magnetic field.
          // Calculate the velocity vector.
          const double v = c1 * sqrt(energy);
          vx = v * kx;
          vy = v * ky;
          vz = v * kz;

          a1 = vx * ex + vy * ey + vz * ez;
          a2 = c2 * (ex * ex + ey * ey + ez * ez);
        }

        if (m_userHandleStep) {
          m_userHandleStep(x, y, z, t, energy, kx, ky, kz, hole);
        }

        // Determine the timestep.
        double dt = 0.;
        /*
        std::cout << "-------------------------------" <<std::endl;
        std::cout << "Initial Time Step : " << dt << " ns"<<std::endl;
        */
        while (1) {
          // Sample the flight time.
          const double r = RndmUniformPos();
          dt += -log(r) * fInv;
          // Calculate the energy after the proposed step.
          if (m_useBfield && bOk) {
            cwt = cos(wb * dt);
            swt = sin(wb * dt);
            newEnergy = std::max(energy + (a1 + a2 * dt) * dt +
                                     a4 * (a3 * (1. - cwt) + vz * swt),
                                 Small);
          } else if (useBandStructure) {
            const double cdt = dt * SpeedOfLight;
            newEnergy = std::max(medium->GetElectronEnergy(
                                     kx + ex * cdt, ky + ey * cdt,
                                     kz + ez * cdt, newVx, newVy, newVz, band),
                                 Small);
          } else {
            newEnergy = std::max(energy + (a1 + a2 * dt) * dt, Small);
          }
          // Get the real collision rate at the updated energy.
          double fReal = medium->GetElectronCollisionRate(newEnergy, band);
          if (fReal <= 0.) {
            std::cerr << hdr << "Got collision rate <= 0 at " << newEnergy
                      << " eV (band " << band << ").\n";
            return false;
          }
          if (fReal > fLim) {
            // Real collision rate is higher than null-collision rate.
            dt += log(r) * fInv;
            // Increase the null collision rate and try again.
            std::cerr << hdr << "Increasing null-collision rate by 5%.\n";
            if (useBandStructure) std::cerr << "    Band " << band << "\n";
            fLim *= 1.05;
            fInv = 1. / fLim;
            continue;
          }
          /*
          std::cout << "fReal : " << fReal << ""<<std::endl;
          std::cout << "fInv  : " << fInv  << ""<<std::endl;
          std::cout << "dt    : " << dt  << " ns"<<std::endl;
          */
          // Check for real or null collision.
          
          if (RndmUniform() <= fReal * fInv){
            //std::cout << "Rndm < fReal*fInv break" <<std::endl;
            break;
          }
          if (m_useNullCollisionSteps) {
            isNullCollision = true;
            break;
          }
        }
        //std::cout << "Electron Time Step : " << dt << " ns"<<std::endl;
        if (!ok) break;

        // Increase the collision counter.
        ++nCollTemp;

        // Update the directions (at instant before collision)
        // and calculate the proposed new position.
        if (m_useBfield && bOk) {
          // Calculate the new velocity.
          newVx = vx + 2. * c2 * ex * dt;
          newVy = vz * swt - a3 * cwt + ez / bmag;
          newVz = vz * cwt + a3 * swt;
          // Normalise and rotate back to the lab frame.
          const double v = sqrt(newVx * newVx + newVy * newVy + newVz * newVz);
          newKx = newVx / v;
          newKy = newVy / v;
          newKz = newVz / v;
          RotateLocal2Global(newKx, newKy, newKz);
          // Calculate the step in coordinate space.
          vx += c2 * ex * dt;
          ky = (vz * (1. - cwt) - a3 * swt) / (wb * dt) + ez / bmag;
          kz = (vz * swt + a3 * (1. - cwt)) / (wb * dt);
          vy = ky;
          vz = kz;
          // Rotate back to the lab frame.
          RotateLocal2Global(vx, vy, vz);
        } else if (useBandStructure) {
          // Update the wave-vector.
          newKx = kx + ex * dt * SpeedOfLight;
          newKy = ky + ey * dt * SpeedOfLight;
          newKz = kz + ez * dt * SpeedOfLight;
          // Average velocity over the step.
          vx = 0.5 * (vx + newVx);
          vy = 0.5 * (vy + newVy);
          vz = 0.5 * (vz + newVz);
        } else {
          // Update the direction.
          a1 = sqrt(energy / newEnergy);
          a2 = 0.5 * c1 * dt / sqrt(newEnergy);
          newKx = kx * a1 + ex * a2;
          newKy = ky * a1 + ey * a2;
          newKz = kz * a1 + ez * a2;

          // Calculate the step in coordinate space.
          a1 = c1 * sqrt(energy);
          a2 = dt * c2;
          vx = kx * a1 + ex * a2;
          vy = ky * a1 + ey * a2;
          vz = kz * a1 + ez * a2;
        }

        double x1 = x + vx * dt;
        double y1 = y + vy * dt;
        double z1 = z + vz * dt;
        double t1 = t + dt;
        // Get the electric field and medium at the proposed new position.
        m_sensor->ElectricField(x1, y1, z1, ex, ey, ez, medium, status);
        if (!hole) {
          ex = -ex;
          ey = -ey;
          ez = -ez;
        }

        // Check if the electron is still inside a drift medium/the drift area.
        if (status != 0 || !m_sensor->IsInArea(x1, y1, z1)) {
          // Try to terminate the drift line close to the boundary (endpoint
          // outside the drift medium/drift area) using iterative bisection.
          Terminate(x, y, z, t, x1, y1, z1, t1);
          if (m_useSignal) {
            const int q = hole ? 1 : -1;
            AddSignalPoint(m_sensor, q, t, t1 - t, x, y, z, x1, y1, z1);
          }
          Update(it, x1, y1, z1, t1, energy, newKx, newKy, newKz, band);
          if (status != 0) {
            (*it).status = StatusLeftDriftMedium;
            if (m_debug)
              PrintStatus(hdr, "left the drift medium", x1, y1, z1, hole);
          } else {
            (*it).status = StatusLeftDriftArea;
            if (m_debug)
              PrintStatus(hdr, "left the drift area", x1, y1, z1, hole);
          }
          AddToEndPoints(*it, hole);
          ok = false;
          break;
        }

        // Check if the electron/hole has crossed a wire.
        double xc = x, yc = y, zc = z;
        double rc = 0.;
        if (m_sensor->CrossedWire(x, y, z, x1, y1, z1, xc, yc, zc, false, rc)) {
          // If switched on, calculated the induced signal over this step.
          if (m_useSignal) {
            const double dx = xc - x;
            const double dy = yc - y;
            const double dz = zc - z;
            dt = sqrt(dx * dx + dy * dy + dz * dz) /
                 sqrt(vx * vx + vy * vy + vz * vz);
            const int q = hole ? 1 : -1;
            AddSignalPoint(m_sensor, q, t, dt, x, y, z, xc, yc, zc);
          }
          Update(it, xc, yc, zc, t + dt, energy, newKx, newKy, newKz, band);
          (*it).status = StatusLeftDriftMedium;
          AddToEndPoints(*it, hole);
          ok = false;
          if (m_debug) PrintStatus(hdr, "hit a wire", x, y, z, hole);
          break;
        }

        // If switched on, calculate the induced signal.
        if (m_useSignal) {
          const int q = hole ? 1 : -1;
          AddSignalPoint(m_sensor, q, t, dt, x, y, z, x1, y1, z1);
        }

        // Update the coordinates.
        x = x1;
        y = y1;
        z = z1;
        t = t1;

        // If switched on, get the magnetic field at the new location.
        if (m_useBfield) {
          m_sensor->MagneticField(x, y, z, bx, by, bz, status);
          const double scale = hole ? Tesla2Internal : -Tesla2Internal;
          bx *= scale;
          by *= scale;
          bz *= scale;
          // Make sure that neither E nor B are zero.
          bmag = sqrt(bx * bx + by * by + bz * bz);
          const double emag2 = ex * ex + ey * ey + ez * ez;
          bOk = (bmag > Small && emag2 > Small);
        }

        if (isNullCollision) {
          energy = newEnergy;
          kx = newKx;
          ky = newKy;
          kz = newKz;
          continue;
        }

        // Get the collision type and parameters.
        int cstype = 0;
        int level = 0;
        int ndxc = 0;
        medium->ElectronCollision(newEnergy, cstype, level, energy, newKx,
                                 newKy, newKz, secondaries, band);
        // If activated, histogram the distance with respect to the
        // last collision.
        if (m_histDistance && !m_distanceHistogramType.empty()) {
          for (const auto& htype : m_distanceHistogramType) {
            if (htype != cstype) continue;
            if (m_debug) {
              std::cout << m_className << "::TransportElectron: Collision type "
                        << cstype << ". Fill distance histogram.\n";
              getchar();
            }
            switch (m_distanceOption) {
              case 'x':
                m_histDistance->Fill((*it).xLast - x);
                break;
              case 'y':
                m_histDistance->Fill((*it).yLast - y);
                break;
              case 'z':
                m_histDistance->Fill((*it).zLast - z);
                break;
              case 'r':
                const double r2 = pow((*it).xLast - x, 2) +
                                  pow((*it).yLast - y, 2) +
                                  pow((*it).zLast - z, 2);
                m_histDistance->Fill(sqrt(r2));
                break;
            }
            (*it).xLast = x;
            (*it).yLast = y;
            (*it).zLast = z;
            break;
          }
        }

        if (m_userHandleCollision) {
          m_userHandleCollision(x, y, z, t, cstype, level, medium, newEnergy,
                                energy, kx, ky, kz, newKx, newKy, newKz);
        }
        switch (cstype) {
          // Elastic collision
          case ElectronCollisionTypeElastic:
            break;
          // Ionising collision
          case ElectronCollisionTypeIonisation:
            if (m_usePlotting && m_plotIonisations) {
              AddLegacyMarker(m_viewer, 'i', x, y, z);
            }
            if (m_userHandleIonisation) {
              m_userHandleIonisation(x, y, z, t, cstype, level, medium);
            }
            for (const auto& secondary : secondaries) {
              if (secondary.type == Particle::Electron) {
                const double esec = std::max(secondary.energy, Small);
                if (m_histSecondary) m_histSecondary->Fill(esec);
                ++m_nElectrons;
                if (!aval) continue;
                if (useBandStructure) {
                  double kxs = 0., kys = 0., kzs = 0.;
                  int bs = -1;
                  medium->GetElectronMomentum(esec, kxs, kys, kzs, bs);
                  AddToStack(x, y, z, t, esec, 0, kxs, kys, kzs, bs, false,
                             stackNew);
                } else {
                  AddToStack(x, y, z, t, esec, 0, false, stackNew);
                }
              } else if (secondary.type == Particle::Hole) {
                const double esec = std::max(secondary.energy, Small);
                ++m_nHoles;
                if (!aval) continue;
                if (useBandStructure) {
                  double kxs = 0., kys = 0., kzs = 0.;
                  int bs = -1;
                  medium->GetElectronMomentum(esec, kxs, kys, kzs, bs);
                  AddToStack(x, y, z, t, esec, 1, kxs, kys, kzs, bs, true,
                             stackNew);
                } else {
                  AddToStack(x, y, z, t, esec, 1, true, stackNew);
                }
              } else if (secondary.type == Particle::Ion) {
                ++m_nIons;
              }
            }
            secondaries.clear();
            if (m_debug) PrintStatus(hdr, "ionised", x, y, z, hole);
            //PrintStatus(hdr, "ionised", x, y, z, hole);
            break;
          // Attachment
          case ElectronCollisionTypeAttachment:
            if (m_usePlotting && m_plotAttachments) {
              AddLegacyMarker(m_viewer, 'a', x, y, z);
            }
            if (m_userHandleAttachment) {
              m_userHandleAttachment(x, y, z, t, cstype, level, medium);
            }
            {
              double ninewkx = 0., ninewky = 0., ninewkz = 0.;
              double niethremal = Small;
              std::array<double, 3> vni = {0., 0., 0.};
              const std::array<double, 3> evec = {ex, ey, ez};
              const std::array<double, 3> bvec = {bx, by, bz};
              if (medium->NegativeIonVelocity(evec[0], evec[1], evec[2],
                                              bvec[0], bvec[1], bvec[2],
                                              vni[0], vni[1], vni[2])) {
                const double vmag = std::sqrt(vni[0] * vni[0] + vni[1] * vni[1] +
                                             vni[2] * vni[2]);
                if (vmag > Small) {
                  ninewkx = vni[0] / vmag;
                  ninewky = vni[1] / vmag;
                  ninewkz = vni[2] / vmag;
                }
              }
              Update(it, x, y, z, t, niethremal, ninewkx, ninewky, ninewkz, band);
              (*it).particle = 2;
              energy = niethremal;
              kx = ninewkx;
              ky = ninewky;
              kz = ninewkz;
            }
            if(m_debug)PrintStatus(hdr, "Attached",x,y,z,hole);
            break;
          // Inelastic collision
          case ElectronCollisionTypeInelastic:
            if (m_userHandleInelastic) {
              m_userHandleInelastic(x, y, z, t, cstype, level, medium);
            }
            break;
          // Excitation
          case ElectronCollisionTypeExcitation:
            if (m_usePlotting && m_plotExcitations) {
              AddLegacyMarker(m_viewer, 'e', x, y, z);
            }
            if (m_userHandleInelastic) {
              m_userHandleInelastic(x, y, z, t, cstype, level, medium);
            }
            for (const auto& sec : secondaries) {
              if (sec.type == Particle::Electron) {
                const double esec = std::max(sec.energy, Small);
                ++m_nElectrons;
                ++m_nIons;
                if (!aval) continue;
                AddToStack(x, y, z, t, esec, 0, false, stackNew);
              } else if (sec.type == Particle::Photon && m_usePhotons &&
                         sec.energy > m_gammaCut) {
                stackPhotons.emplace_back(std::make_pair(t + sec.time, sec.energy));
              }
            }

            // Transport the photons (if any)
            if (aval) {
              for (const auto& ph : stackPhotons) {
                TransportPhoton(x, y, z, ph.first, ph.second, stackNew);
              }
            }
            break;
          // Super-elastic collision
          case ElectronCollisionTypeSuperelastic:
            break;
          // Virtual/null collision
          case ElectronCollisionTypeVirtual:
            break;
          // Acoustic phonon scattering (intravalley)
          case ElectronCollisionTypeAcousticPhonon:
            break;
          // Optical phonon scattering (intravalley)
          case ElectronCollisionTypeOpticalPhonon:
            break;
          // Intervalley scattering (phonon assisted)
          case ElectronCollisionTypeIntervalleyG:
          case ElectronCollisionTypeIntervalleyF:
          case ElectronCollisionTypeInterbandXL:
          case ElectronCollisionTypeInterbandXG:
          case ElectronCollisionTypeInterbandLG:
            break;
          // Coulomb scattering
          case ElectronCollisionTypeImpurity:
            break;
          default:
            std::cerr << hdr << "Unknown collision type.\n";
            ok = false;
            break;
        }

        // Continue with the next electron/hole?
        if (!ok || nCollTemp > m_nCollSkip ||
            cstype == ElectronCollisionTypeIonisation ||
            (m_plotExcitations && cstype == ElectronCollisionTypeExcitation) ||
            (m_plotAttachments && cstype == ElectronCollisionTypeAttachment)) {
          break;
        }
        kx = newKx;
        ky = newKy;
        kz = newKz;
      }

      if (!ok) continue;

      if (!useBandStructure) {
        // Normalise the direction vector.
        const double k = sqrt(kx * kx + ky * ky + kz * kz);
        kx /= k;
        ky /= k;
        kz /= k;
      }
      // Update the stack.
      Update(it, x, y, z, t, energy, kx, ky, kz, band);
      // Add a new point to the drift line (if enabled).
      if (m_useDriftLines) {
        point newPoint;
        newPoint.x = x;
        newPoint.y = y;
        newPoint.z = z;
        newPoint.t = t;
        if((*it).particle==2){
          newPoint.changePoint = 2;
        }else{
          newPoint.changePoint = 0;
        }
        (*it).driftLine.push_back(std::move(newPoint));
      }
    }
  }

  // Calculate the induced charge.
  if (m_useInducedCharge) {
    for (const auto& ep : m_endpointsElectrons) {
      m_sensor->AddInducedCharge(-1, ep.x0, ep.y0, ep.z0, ep.x, ep.y, ep.z);
    }
    for (const auto& ep : m_endpointsHoles) {
      m_sensor->AddInducedCharge(+1, ep.x0, ep.y0, ep.z0, ep.x, ep.y, ep.z);
    }
  }

  // Plot the drift paths and photon tracks.
  if (m_usePlotting) {
    // Electrons
    /*
    const unsigned int nElectronEndpoints = m_endpointsElectrons.size();
    for (unsigned int i = 0; i < nElectronEndpoints; ++i) {
      const int np = GetNumberOfElectronDriftLinePoints(i);
      int jL;
      if (np <= 0) continue;
      const Electron& p = m_endpointsElectrons[i];
      m_viewer->NewElectronDriftLine(np, jL, p.x0, p.y0, p.z0);
      for (int jP = np; jP--;) {
        double x = 0., y = 0., z = 0., t = 0.;
        int changePoint = 0;
        GetElectronDriftLinePoint(x, y, z, t, changePoint, jP, i);
        m_viewer->SetDriftLinePoint(jL, jP, x, y, z);
      }
    }
    */
    // Electrons & NegativeIons
    const unsigned int nNegativeIonEndPoints = m_endpointsElectrons.size();
    for(unsigned int i = 0; i < nNegativeIonEndPoints; ++i){
      const int np = GetNumberOfElectronDriftLinePoints(i);
      std::vector<double> v_x, v_y, v_z;
      int jL;
      int LastFlag = 2; // Electorn=1 NegativeIon=2
      const Electron& p = m_endpointsElectrons[i];
      bool firstFlag = true;
      for(int jP = 0; jP < np ; jP++){
        double x = 0., y = 0., z = 0., t = 0.;
        int cPoint = 0;
        GetElectronDriftLinePoint(x, y, z, t, cPoint, jP, i);
        if(jP+1==np)cPoint = LastFlag;
        v_x.push_back(x);
        v_y.push_back(y);
        v_z.push_back(z);
        if(cPoint==2){
          /*
          jP++;
          GetElectronDriftLinePoint(x, y, z, t, cPoint, jP, i);
          v_x.push_back(x);
          v_y.push_back(y);
          v_z.push_back(z);
          */
          //std::cout << "ElectronDriftLine::Draw" <<std::endl;
          if(firstFlag){
            jL = m_viewer->NewDriftLine(Particle::Electron, v_x.size(), p.x0, p.y0, p.z0);
            firstFlag = false;
          }else{
            jL = m_viewer->NewDriftLine(Particle::Electron, v_x.size(), v_x[0], v_y[0], v_z[0]);
          }
          for(unsigned int po = 0; po < v_x.size(); po++){
            m_viewer->SetDriftLinePoint(jL,po,v_x[po],v_y[po],v_z[po]);
          }
          v_x.clear();
          v_y.clear();
          v_z.clear();
          v_x.push_back(x);
          v_y.push_back(y);
          v_z.push_back(z);
          LastFlag=1;
        }else if(cPoint==1){
          /*
          jP++;
          GetElectronDriftLinePoint(x, y, z, t, cPoint, jP, i);
          v_x.push_back(x);
          v_y.push_back(y);
          v_z.push_back(z);
          */
          //std::cout << "NegativeIonDriftLine::Draw" <<std::endl;
          if(firstFlag){
            jL = m_viewer->NewDriftLine(Particle::NegativeIon, v_x.size(), p.x0, p.y0, p.z0);
            firstFlag = false;
          }else{
            jL = m_viewer->NewDriftLine(Particle::NegativeIon, v_x.size(), v_x[0], v_y[0], v_z[0]);
          }
          for(unsigned int po = 0; po < v_x.size(); po++){
            m_viewer->SetDriftLinePoint(jL,po,v_x[po],v_y[po],v_z[po]);
          }
          v_x.clear();
          v_y.clear();
          v_z.clear();
          v_x.push_back(x);
          v_y.push_back(y);
          v_z.push_back(z);
          LastFlag=2;
        }else{
        }
      }
    }
    // Holes
    const unsigned int nHoleEndpoints = m_endpointsHoles.size();
    for (unsigned int i = 0; i < nHoleEndpoints; ++i) {
      const int np = GetNumberOfHoleDriftLinePoints(i);
      int jL;
      if (np <= 0) continue;
      const Electron& p = m_endpointsHoles[i];
      jL = m_viewer->NewDriftLine(Particle::Hole, np, p.x0, p.y0, p.z0);
      for (int jP = np; jP--;) {
        double x = 0., y = 0., z = 0., t = 0.;
        GetHoleDriftLinePoint(x, y, z, t, jP, i);
        m_viewer->SetDriftLinePoint(jL, jP, x, y, z);
      }
    }
    // Photons
    for (const auto& ph : m_photons) {
      m_viewer->AddPhoton(ph.x0, ph.y0, ph.z0, ph.x1, ph.y1, ph.z1);
    }
  }
  return true;
}

void AvalancheNIMicroscopic::TransportPhoton(const double x0, const double y0,
                                           const double z0, const double t0,
                                           const double e0,
                                           std::vector<Electron>& stack) {
  // Make sure that the sensor is defined.
  if (!m_sensor) {
    std::cerr << m_className << "::TransportPhoton: Sensor is not defined.\n";
    return;
  }

  // Make sure that the starting point is inside a medium.
  Medium* medium = m_sensor->GetMedium(x0, y0, z0);
  if (!medium) {
    std::cerr << m_className << "::TransportPhoton:\n"
              << "    No medium at initial position.\n";
    return;
  }

  // Make sure that the medium is "driftable" and microscopic.
  if (!medium->IsDriftable() || !medium->IsMicroscopic()) {
    std::cerr << m_className << "::TransportPhoton:\n"
              << "    Medium at initial position does not provide "
              << " microscopic tracking data.\n";
    return;
  }

  // Get the id number of the drift medium.
  int id = medium->GetId();

  // Position
  double x = x0, y = y0, z = z0;
  double t = t0;
  // Initial direction (randomised).
  double dx = 0., dy = 0., dz = 0.;
  RndmDirection(dx, dy, dz);
  // Energy
  double e = e0;

  // Photon collision rate
  double f = medium->GetPhotonCollisionRate(e);
  if (f <= 0.) return;
  // Timestep
  double dt = -log(RndmUniformPos()) / f;
  t += dt;
  dt *= SpeedOfLight;
  x += dt * dx;
  y += dt * dy;
  z += dt * dz;

  // Check if the photon is still inside a medium.
  medium = m_sensor->GetMedium(x, y, z);
  if (!medium || medium->GetId() != id) {
    // Try to terminate the photon track close to the boundary
    // by means of iterative bisection.
    dx *= dt;
    dy *= dt;
    dz *= dt;
    x -= dx;
    y -= dy;
    z -= dz;
    double delta = sqrt(dx * dx + dy * dy + dz * dz);
    if (delta > 0) {
      dx /= delta;
      dy /= delta;
      dz /= delta;
    }
    // Mid-point
    double xM = x, yM = y, zM = z;
    while (delta > BoundaryDistance) {
      delta *= 0.5;
      dt *= 0.5;
      xM = x + delta * dx;
      yM = y + delta * dy;
      zM = z + delta * dz;
      // Check if the mid-point is inside the drift medium.
      medium = m_sensor->GetMedium(xM, yM, zM);
      if (medium && medium->GetId() == id) {
        x = xM;
        y = yM;
        z = zM;
        t += dt;
      }
    }
    photon newPhoton;
    newPhoton.x0 = x0;
    newPhoton.y0 = y0;
    newPhoton.z0 = z0;
    newPhoton.x1 = x;
    newPhoton.y1 = y;
    newPhoton.z1 = z;
    newPhoton.energy = e0;
    newPhoton.status = StatusLeftDriftMedium;
    m_photons.push_back(std::move(newPhoton));
    return;
  }

  int type, level;
  double e1;
  double ctheta = 0.;
  int nsec = 0;
  double esec = 0.;
  std::vector<Medium::Secondary> secondaries;
  if (!medium->PhotonCollision(e, type, level, e1, ctheta, secondaries))
    return;
  nsec = static_cast<int>(secondaries.size());
  esec = 0.;
  if (nsec > 0) {
    for (const auto& sec : secondaries) {
      if (sec.type == Particle::Electron) esec = std::max(esec, sec.energy);
    }
  }

  if (type == PhotonCollisionTypeIonisation) {
    // Add the secondary electron (random direction) to the stack.
    if (m_sizeCut == 0 || stack.size() < m_sizeCut) {
      //AddToStack(x, y, z, t, std::max(esec, Small), false, stack);
      AddToStack(x, y, z, t, std::max(esec, Small), 0, false, stack);
    }
    // Increment the electron and ion counters.
    ++m_nElectrons;
    ++m_nIons;
  } else if (type == PhotonCollisionTypeExcitation) {
    double tdx = 0.;
    double sdx = 0.;
    int typedx = 0;
    std::vector<double> tPhotons;
    std::vector<double> ePhotons;
    for (int j = nsec; j--;) {
      if (m_usePhotons) {
        // The modern MediumMagboltz API emits secondary products through the
        // standard collision secondaries container, so no legacy fetch is needed here.
      }
    }
    // Transport the photons (if any).
    const int nSizePhotons = tPhotons.size();
    for (int k = nSizePhotons; k--;) {
      TransportPhoton(x, y, z, tPhotons[k], ePhotons[k], stack);
    }
  }

  photon newPhoton;
  newPhoton.x0 = x0;
  newPhoton.y0 = y0;
  newPhoton.z0 = z0;
  newPhoton.x1 = x;
  newPhoton.y1 = y;
  newPhoton.z1 = z;
  newPhoton.energy = e0;
  newPhoton.status = -2;
  m_photons.push_back(std::move(newPhoton));
}


//---------------------------------------------------------
// T.Shimada Develop
void AvalancheNIMicroscopic::InputDetachCrossSectionData(std::string filename)
{

  m_DetachData_xsec.clear();
  m_DetachData_xene.clear();

  m_DetachCrossSectionPoint = 0;
  std::ifstream ifs(filename.c_str(),std::ios::in);
  double this_cross;
  double this_ene;
  while(ifs >> this_ene >> this_cross){
    m_DetachCrossSectionPoint++;
    m_DetachData_xsec.push_back(this_cross);
    m_DetachData_xene.push_back(this_ene);
  }
  if(m_DetachData_xsec.size()!=m_DetachData_xene.size() || m_DetachData_xsec.size()!=m_DetachCrossSectionPoint){
    std::cerr << " input file faild... " <<std::endl;
    return;
  }
  if(true){//Debug
    //std::cerr << "Detach Data Point : " << m_DetachCrossSectionPoint <<std::endl;
    for(int i=0;i<m_DetachCrossSectionPoint;i++){
      std::cerr << "Energy : " << m_DetachData_xene.at(i) << ", Cross-Section : " << m_DetachData_xsec.at(i) <<std::endl;
    }
  }

}

double AvalancheNIMicroscopic::GetDetachCrossSection(double energy)
{
  if(m_DetachCrossSectionPoint==0){
    std::cerr << "Not Input DetachCrossSection File" << std::endl;
    return -1;
  }
  double x_sec; //[*1e20 m^2]
  //Get Maximum & Minumum
  double Emin = *std::min_element(m_DetachData_xene.begin(), m_DetachData_xene.end());
  double Emax = *std::max_element(m_DetachData_xene.begin(), m_DetachData_xene.end());
  //std::cerr << "Energy : [ " << Emin << " ~ " << Emax << " ]" <<std::endl;
  //Near
  if(false){
    double dene_min = 1e20;//eV
    if(energy < Emin || Emax < energy){
      return -1;
    }else{
      for(int data=0;data<m_DetachCrossSectionPoint;data++){
        double this_xsec = m_DetachData_xsec.at(data);
        double this_xene = m_DetachData_xene.at(data);
        if(dene_min > abs(energy-this_xene)){
          dene_min = abs(energy-this_xene);
          x_sec = this_xsec;
        }
      }
    }
    //std::cerr << "x_sec : " << x_sec <<std::endl;
    return x_sec;
  }
  //Liner
  if(true){
    if(energy < Emin || Emax < energy){
      return -1;
    }else{
      double lowEne=Emin,highEne=Emax;
      double lowCro,highCro;
      double this_xsec,this_xene;
      for(int data=0;data<m_DetachCrossSectionPoint;data++){
        this_xsec = m_DetachData_xsec.at(data);
        this_xene = m_DetachData_xene.at(data);
        highEne = this_xene;
        highCro = this_xsec;
        //std::cerr << "Low vs High : " << lowEne << ", " << highEne <<std::endl;
        if(lowEne <= energy && energy < highEne){
          x_sec = lowCro + (highCro-lowCro)*(energy-lowEne)/(highEne-lowEne);
          break;
        }
        lowEne = this_xene;
        lowCro = this_xsec;
      }
    }
    //std::cerr << "x_sec : " << x_sec <<std::endl;
    return x_sec;
  }
}

double AvalancheNIMicroscopic::ComputeLambda(double pressure, double x_sec)
{ 
  //pressure [Torr]
  double Temp = 300.; // K
  double ndensity; // *1e20 m^3
  double lambda; // m
  ndensity = pressure*133.32 / (1.38066*1e-23*Temp) /1e20; // *1e20 m^3
  lambda = 1./ (ndensity * x_sec); // 1 / ( 1e20 * m^3 * 1e-20 * m^2) -> m
  //cerr << "x-sec : " << x_sec << ", lambda : " << lambda*100 << endl;
  return lambda*100; // [cm]
}

void AvalancheNIMicroscopic::SetDetachModel(int model_num)
{
  DetachmentModelNo = model_num;
}

/*
int AvalancheNIMicroscopic::DetachmentNegativeIon(const double step, Medium* medium,
                                       const std::array<double, 3>& e,
                                       const std::array<double, 3>& v,
                                       double inputZ, int& status)
{
  double VelocityAbs = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])*1e-1/1e-9 / (3.0*1e8); // [cm*1e-1/ns*1e-9] / [m/s]  -> no unit
  double EfieldAbs   = sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]); // V/cm
  double nionMW   = inputZ;//SF6:146 F:19
  double nionMass = nionMW*931.5*1e6; //eV
  double energyIon   = 1./2.*nionMass*VelocityAbs*VelocityAbs;
  if(false){
    std::cerr << "Negative Ion Velocity       : " << sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])*1e6 << " cm/ms " <<std::endl;
    std::cerr << "Negative Ion Electric Field : " << sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]) << " V/cm " <<std::endl;
    std::cerr << "Negative Ion Energy         : " << energyIon << " eV " <<std::endl;
  }
  //set detachment model
  //bool ThresholdModel = true;
  //bool CrossSectionModel = false;
  // Cross-Section Model
  if(DetachmentModelNo==0){
    // input x-section
    double Torr = medium->GetPressure();
    double dx = step;
    double prob=0;
    double this_xsec = GetDetachCrossSection(energyIon);
    double this_lambda = ComputeLambda(Torr,this_xsec);
    if(this_lambda <= 0) return -1;
    
    prob = 1. - exp(-dx/this_lambda);
    //std::cerr << "probability : " << prob <<std::endl;
    if(RndmUniform() < prob ){ //40kV/cm
      status = StatusSF6EDetached;
      return 1;
    }
  }
  //Threshold model (SF6- -> SF6 + e-)
  if(DetachmentModelNo==1){
    double threshold = 40000; // V/cm
    double sigma = 10; // %
    double prob = 0.5*(1.+erf((EfieldAbs-threshold)/(threshold*sigma/100.)));
    //std::cerr << "prob = " << prob << " , Efield = " << EfieldAbs << " V/cm" <<std::endl;
    if(RndmUniform() < prob ){ //40kV/cm
      status = StatusSF6EDetached;
      return 1;
    }
  }
  return -1;

}
*/
int AvalancheNIMicroscopic::GetField(const std::array<double, 3>& x,
                                     std::array<double, 3>& e, std::array<double, 3>& b,
                                     Medium*& medium){
  e.fill(0.);
  b.fill(0.);
  // Get the electric field.
  int status = 0;
  m_sensor->ElectricField(x[0], x[1], x[2], e[0], e[1], e[2], medium, status);
  // Make sure the point is inside a drift medium.
  if (status != 0 || !medium) return StatusLeftDriftMedium;
  // Make sure the point is inside the drift area.
  if (!m_sensor->IsInArea(x[0], x[1], x[2])) return StatusLeftDriftArea;

  // Get the magnetic field, if requested.
  if (m_useBfield) {
    m_sensor->MagneticField(x[0], x[1], x[2], b[0], b[1], b[2], status);
    for (unsigned int k = 0; k < 3; ++k) b[k] *= Tesla2Internal;
  }
  return 0;
}

bool AvalancheNIMicroscopic::GetVelocity(const int type, Medium* medium, 
                              const std::array<double, 3>& x,
                              const std::array<double, 3>& e,
                              const std::array<double, 3>& b,
                              std::array<double, 3>& v){
  v.fill(0.);
  bool ok = false;
  if (type < 0) {
    ok = medium->ElectronVelocity(e[0], e[1], e[2], b[0], b[1], b[2], 
                                  v[0], v[1], v[2]);
  } else if (type == 1) {
    ok = medium->HoleVelocity(e[0], e[1], e[2], b[0], b[1], b[2], 
                              v[0], v[1], v[2]);
  } else if (type == 2) {
    ok = medium->IonVelocity(e[0], e[1], e[2], b[0], b[1], b[2], 
                             v[0], v[1], v[2]);
  } else if (type == 3) {
    ok = medium->NegativeIonVelocity(e[0], e[1], e[2], b[0], b[1], b[2], 
                             v[0], v[1], v[2]);
  }
  if (!ok) {
    //PrintError("GetVelocity", "velocity", type, x);
    return false;
  }
  if (m_debug) {
    //std::cout << m_className << "::GetVelocity: Velocity at "
    //          << PrintVec(x) << " = " << PrintVec(v) << "\n";
  }
  return true;
}


void AvalancheNIMicroscopic::InputNegativeIonDiffusionData(std::string filename_dl, std::string filename_dt)
{

  double this_e;
  double this_d;
  
  // Longitudial
  m_DiffusionData_edl.clear();
  m_DiffusionData_dl.clear();
  std::ifstream ifs_dl(filename_dl.c_str(),std::ios::in);
  while(ifs_dl>>this_e>>this_d){
    m_DiffusionData_edl.push_back(this_e);
    m_DiffusionData_dl.push_back(this_d);
  }

  // Transverse
  m_DiffusionData_edt.clear();
  m_DiffusionData_dt.clear();
  std::ifstream ifs_dt(filename_dt.c_str(),std::ios::in);
  while(ifs_dt>>this_e>>this_d){
    m_DiffusionData_edt.push_back(this_e);
    m_DiffusionData_dt.push_back(this_d);
  }

  if(m_DiffusionData_edl.size()!=m_DiffusionData_dl.size() || m_DiffusionData_edt.size()!=m_DiffusionData_dt.size() ){
    std::cerr << " input file faild... " <<std::endl;
    return;
  }
  if(true){//Debug
    //std::cerr << "Detach Data Point : " << m_DetachCrossSectionPoint <<std::endl;
    for(int i=0;i<m_DiffusionData_edl.size();i++){
      std::cerr << "E (V/cm) : " << m_DiffusionData_edl.at(i) << 
                   ", Longitudial : " << m_DiffusionData_dl.at(i) <<std::endl;
    }
    for(int i=0;i<m_DiffusionData_edt.size();i++){
      std::cerr << "E (V/cm) : " << m_DiffusionData_edt.at(i) << 
                   ", Transverse : " << m_DiffusionData_dt.at(i) <<std::endl;
    }
  }

  if(m_DiffusionData_edl.size()!=0 && m_DiffusionData_edl.size()!=0){
    m_useDiffusionData = true;
  }else{
    std::cout << "Negative Ion Diffusion Input File Faild..." <<std::endl;
    m_useDiffusionData = false;
  }
  return;

}

bool AvalancheNIMicroscopic::GetNegativeIonDiffusion(const double e, const double temp, double& dl, double& dt)
{

  if(m_useDiffusionData){
    
    // Longitudial
    double EDLmin  = *std::min_element(m_DiffusionData_edl.begin(), m_DiffusionData_edl.end());
    double EDLmax  = *std::max_element(m_DiffusionData_edl.begin(), m_DiffusionData_edl.end());
    double DLmin = *std::min_element(m_DiffusionData_dl.begin(), m_DiffusionData_dl.end());
    double DLmax = *std::max_element(m_DiffusionData_dl.begin(), m_DiffusionData_dl.end());
    if(e < EDLmin || EDLmax < e){
      dl = sqrt(2. * BoltzmannConstant * temp / e);
    }else{
      double lowEDL = EDLmin, highEDL = EDLmax;
      double lowDL,highDL;
      double this_edl,this_dl;
      for(int i=0;i<m_DiffusionData_edl.size();i++){
        this_edl = m_DiffusionData_edl.at(i);
        this_dl  = m_DiffusionData_dl.at(i);
        highEDL = this_edl;
        highDL  = this_dl;
        if(lowEDL <= e && e < highEDL){
          dl = lowDL + (highDL-lowDL)*(e-lowEDL)/(highEDL-lowEDL);
          break;
        }
        lowEDL = this_edl;
        lowDL  = this_dl;
      }
    }

    // Transverse
    double EDTmin  = *std::min_element(m_DiffusionData_edt.begin(), m_DiffusionData_edt.end());
    double EDTmax  = *std::max_element(m_DiffusionData_edt.begin(), m_DiffusionData_edt.end());
    double DTmin = *std::min_element(m_DiffusionData_dt.begin(), m_DiffusionData_dt.end());
    double DTmax = *std::max_element(m_DiffusionData_dt.begin(), m_DiffusionData_dt.end());
    if(e < EDTmin || EDTmax < e){
      dt = sqrt(2. * BoltzmannConstant * temp / e);
    }else{
      double lowEDT = EDTmin, highEDT = EDTmax;
      double lowDT,highDT;
      double this_edt,this_dt;
      for(int i=0;i<m_DiffusionData_edt.size();i++){
        this_edt = m_DiffusionData_edt.at(i);
        this_dt  = m_DiffusionData_dt.at(i);
        highEDT = this_edt;
        highDT  = this_dt;
        if(lowEDT <= e && e < highEDT){
          dt = lowDT + (highDT-lowDT)*(e-lowEDT)/(highEDT-lowEDT);
          break;
        }
        lowEDT = this_edt;
        lowDT  = this_dt;
      }
    }
    return true;
  }
  
  return false;
}

bool AvalancheNIMicroscopic::AddNIDiffusion(const int type, Medium* medium,
                                          const double step, 
                                          std::array<double, 3>& x,
                                          const std::array<double, 3>& v,
                                          const std::array<double, 3>& e,
                                          const std::array<double, 3>& b) {
  bool ok = false;
  //int type = 3;
  double dl = 0., dt = 0.;
  
  double pressure = medium->GetPressure();
  double temp = medium->GetTemperature();
  double e_abs = sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]);
  if(!GetNegativeIonDiffusion(e_abs,temp,dl,dt)){
    const double d = sqrt(2. * BoltzmannConstant * temp / sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]));
    dl = d;
    dt = d;
  }
  
  // Debug
  //std::cout << "e  : " << e_abs <<std::endl;
  //std::cout << "dl : " << dl <<std::endl;
  //std::cout << "dt : " << dt <<std::endl;

  // Scale Pressure
  dl = dl*sqrt(760./pressure);
  dt = dt*sqrt(760./pressure);

  //ok = medium->IonDiffusion(e[0], e[1], e[2], b[0], b[1], b[2], dl, dt);


  // Draw a random diffusion direction in the particle frame.
  const std::array<double, 3> d = {step * RndmGaussian(0., dl), 
                                   step * RndmGaussian(0., dt),
                                   step * RndmGaussian(0., dt)};
  // Compute the rotation angles to align diffusion and drift velocity vectors.
  const double vt = sqrt(v[0] * v[0] + v[1] * v[1]);
  const double phi = vt > Small ? atan2(v[1], v[0]) : 0.;
  const double theta = vt > Small ? atan2(v[2], vt) : v[2] < 0. ? -HalfPi : HalfPi;
  const double cphi = cos(phi);
  const double sphi = sin(phi);
  const double ctheta = cos(theta);
  const double stheta = sin(theta);

  x[0] += cphi * ctheta * d[0] - sphi * d[1] - cphi * stheta * d[2];
  x[1] += sphi * ctheta * d[0] + cphi * d[1] - sphi * stheta * d[2];
  x[2] +=        stheta * d[0] +                      ctheta * d[2];
  return true;
}

/*
bool AvalancheNIMicroscopic::AddNIDiffusion(const int type, Medium* medium,
                                          const double step, 
                                          std::array<double, 3>& x,
                                          const std::array<double, 3>& v,
                                          const std::array<double, 3>& e,
                                          const std::array<double, 3>& b) {
  bool ok = false;
  //int type = 3;
  double dl = 0., dt = 0.;
  if (type < 0) {
    ok = medium->ElectronDiffusion(e[0], e[1], e[2], b[0], b[1], b[2], dl, dt);
  } else if (type == 1) {
    ok = medium->HoleDiffusion(e[0], e[1], e[2], b[0], b[1], b[2], dl, dt);
  } else if (type == 2) {
    ok = medium->IonDiffusion(e[0], e[1], e[2], b[0], b[1], b[2], dl, dt);
  } else if (type == 3) {
    ok = medium->IonDiffusion(e[0], e[1], e[2], b[0], b[1], b[2], dl, dt);
  }
  if (!ok) {
    //PrintError("AddDiffusion", "diffusion", type, x);
    return false;
  }

  // Draw a random diffusion direction in the particle frame.
  const std::array<double, 3> d = {step * RndmGaussian(0., dl), 
                                   step * RndmGaussian(0., dt),
                                   step * RndmGaussian(0., dt)};
  if (m_debug) {
    //std::cout << m_className << "::AddDiffusion: Adding diffusion step " 
    //          << PrintVec(d) << "\n";
  }
  // Compute the rotation angles to align diffusion and drift velocity vectors.
  const double vt = sqrt(v[0] * v[0] + v[1] * v[1]);
  const double phi = vt > Small ? atan2(v[1], v[0]) : 0.;
  const double theta = vt > Small ? atan2(v[2], vt) : v[2] < 0. ? -HalfPi : HalfPi;
  const double cphi = cos(phi);
  const double sphi = sin(phi);
  const double ctheta = cos(theta);
  const double stheta = sin(theta);

  x[0] += cphi * ctheta * d[0] - sphi * d[1] - cphi * stheta * d[2];
  x[1] += sphi * ctheta * d[0] + cphi * d[1] - sphi * stheta * d[2];
  x[2] +=        stheta * d[0] +                      ctheta * d[2];
  return true;
}
*/

//---------------------------------------------------------

void AvalancheNIMicroscopic::ComputeRotationMatrix(
    const double bx, const double by, const double bz, const double bmag,
    const double ex, const double ey, const double ez) {
  // Adopting the Magboltz convention, the stepping is performed
  // in a coordinate system with the B field along the x axis
  // and the electric field at an angle btheta in the x-z plane.

  // Calculate the first rotation matrix (to align B with x axis).
  const double bt = by * by + bz * bz;
  if (bt < Small) {
    // B field is already along axis.
    m_rb11 = m_rb22 = m_rb33 = 1.;
    m_rb12 = m_rb13 = m_rb21 = m_rb23 = m_rb31 = m_rb32 = 0.;
  } else {
    const double btInv = 1. / bt;
    m_rb11 = bx / bmag;
    m_rb12 = by / bmag;
    m_rb21 = -m_rb12;
    m_rb13 = bz / bmag;
    m_rb31 = -m_rb13;
    m_rb22 = (m_rb11 * by * by + bz * bz) * btInv;
    m_rb33 = (m_rb11 * bz * bz + by * by) * btInv;
    m_rb23 = m_rb32 = (m_rb11 - 1.) * by * bz * btInv;
  }
  // Calculate the second rotation matrix (rotation around x axis).
  const double fy = m_rb21 * ex + m_rb22 * ey + m_rb23 * ez;
  const double fz = m_rb31 * ex + m_rb32 * ey + m_rb33 * ez;
  const double ft = sqrt(fy * fy + fz * fz);
  if (ft < Small) {
    // E and B field are parallel.
    m_rx22 = m_rx33 = 1.;
    m_rx23 = m_rx32 = 0.;
  } else {
    m_rx22 = m_rx33 = fz / ft;
    m_rx23 = -fy / ft;
    m_rx32 = -m_rx23;
  }
}

void AvalancheNIMicroscopic::RotateGlobal2Local(double& dx, double& dy,
                                              double& dz) const {
  const double dx1 = m_rb11 * dx + m_rb12 * dy + m_rb13 * dz;
  const double dy1 = m_rb21 * dx + m_rb22 * dy + m_rb23 * dz;
  const double dz1 = m_rb31 * dx + m_rb32 * dy + m_rb33 * dz;

  dx = dx1;
  dy = m_rx22 * dy1 + m_rx23 * dz1;
  dz = m_rx32 * dy1 + m_rx33 * dz1;
}

void AvalancheNIMicroscopic::RotateLocal2Global(double& dx, double& dy,
                                              double& dz) const {
  const double dx1 = dx;
  const double dy1 = m_rx22 * dy + m_rx32 * dz;
  const double dz1 = m_rx23 * dy + m_rx33 * dz;

  dx = m_rb11 * dx1 + m_rb21 * dy1 + m_rb31 * dz1;
  dy = m_rb12 * dx1 + m_rb22 * dy1 + m_rb32 * dz1;
  dz = m_rb13 * dx1 + m_rb23 * dy1 + m_rb33 * dz1;
}

void AvalancheNIMicroscopic::Update(std::vector<Electron>::iterator it,
                                  const double x, const double y,
                                  const double z, const double t,
                                  const double energy, const double kx,
                                  const double ky, const double kz,
                                  const int band) {
  (*it).x = x;
  (*it).y = y;
  (*it).z = z;
  (*it).t = t;
  (*it).energy = energy;
  (*it).kx = kx;
  (*it).ky = ky;
  (*it).kz = kz;
  (*it).band = band;
}

void AvalancheNIMicroscopic::AddToStack(const double x, const double y,
                                      const double z, const double t,
                                      const double energy, const int particle, const bool hole,
                                      std::vector<Electron>& container) const {
  // Randomise the direction.
  double dx = 0., dy = 0., dz = 1.;
  RndmDirection(dx, dy, dz);
  //AddToStack(x, y, z, t, energy, dx, dy, dz, 0, hole, container);
  AddToStack(x, y, z, t, energy, particle, dx, dy, dz, 0, hole, container);
}

void AvalancheNIMicroscopic::AddToStack(const double x, const double y,
                                      const double z, const double t,
                                      const double energy, const int particle, 
                                      const double dx, const double dy, const double dz,
                                      const int band, const bool hole,
                                      std::vector<Electron>& container) const {
  Electron electron;
  electron.status = 0;
  electron.hole = hole;
  electron.x0 = x;
  electron.y0 = y;
  electron.z0 = z;
  electron.t0 = t;
  electron.e0 = energy;
  electron.x = x;
  electron.y = y;
  electron.z = z;
  electron.t = t;
  electron.energy = energy;
  electron.particle = particle;
  electron.kx = dx;
  electron.ky = dy;
  electron.kz = dz;
  electron.band = band;
  // Previous coordinates for distance histogramming.
  electron.xLast = x;
  electron.yLast = y;
  electron.zLast = z;
  electron.driftLine.reserve(1000);
  container.push_back(std::move(electron));
}

void AvalancheNIMicroscopic::Terminate(double x0, double y0, double z0, double t0,
                                     double& x1, double& y1, double& z1,
                                     double& t1) {
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const double dz = z1 - z0;
  double d = sqrt(dx * dx + dy * dy + dz * dz);
  while (d > BoundaryDistance) {
    d *= 0.5;
    const double xm = 0.5 * (x0 + x1);
    const double ym = 0.5 * (y0 + y1);
    const double zm = 0.5 * (z0 + z1);
    const double tm = 0.5 * (t0 + t1);
    // Check if the mid-point is inside the drift medium.
    double ex = 0., ey = 0., ez = 0.;
    Medium* medium = nullptr;
    int status = 0;
    m_sensor->ElectricField(xm, ym, zm, ex, ey, ez, medium, status);
    if (status == 0 && m_sensor->IsInArea(xm, ym, zm)) {
      x0 = xm;
      y0 = ym;
      z0 = zm;
      t0 = tm;
    } else {
      x1 = xm;
      y1 = ym;
      z1 = zm;
      t1 = tm;
    }
  }
}
}