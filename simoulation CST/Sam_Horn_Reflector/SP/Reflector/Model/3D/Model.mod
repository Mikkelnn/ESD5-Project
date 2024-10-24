'# MWS Version: Version 2024.0 - Sep 01 2023 - ACIS 33.0.1 -

'# length = cm
'# frequency = GHz
'# time = ns
'# frequency range: fmin = 9 fmax = 10
'# created = '[VERSION]2024.0|33.0.1|20230901[/VERSION]


'@ define units

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With Units 
     .SetUnit "Length", "cm"
     .SetUnit "Temperature", "degC"
     .SetUnit "Voltage", "V"
     .SetUnit "Current", "A"
     .SetUnit "Resistance", "Ohm"
     .SetUnit "Conductance", "S"
     .SetUnit "Capacitance", "pF"
     .SetUnit "Inductance", "nH"
     .SetUnit "Frequency", "GHz"
     .SetUnit "Time", "ns"
     .SetResultUnit "frequency", "frequency", "" 
End With

'@ internal simulation project settings: 1

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
SetWCSFromReferenceBlockInAssembly "parabolicwork_1"

'@ transform local coordinates

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
TransformCurrentWCS "parabolicwork_1", False

'@ import external project: ..\..\Model\DS\Block\0\B2029407574\parabolicwork.cst

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
StartSubProject 

With Material 
     .Reset 
     .Name "PEC" 
     .Folder "parabolicwork_1" 
     .Rho "0"
     .ThermalType "PTC"
     .MechanicsType "Unused"
     .IntrinsicCarrierDensity "0"
     .FrqType "all"
     .Type "Pec"
     .MaterialUnit "Frequency", "Hz"
     .MaterialUnit "Geometry", "m"
     .MaterialUnit "Time", "s"
     .MaterialUnit "Temperature", "K"
     .Epsilon "1"
     .Mu "1"
     .ReferenceCoordSystem "Global"
     .CoordSystemType "Cartesian"
     .NLAnisotropy "False"
     .NLAStackingFactor "1"
     .NLADirectionX "1"
     .NLADirectionY "0"
     .NLADirectionZ "0"
     .Colour "0.8", "0.8", "0.8" 
     .Wireframe "False" 
     .Reflection "True" 
     .Allowoutline "True" 
     .Transparentoutline "False" 
     .Transparency "0" 
     .Create
End With 

With SAT
     .Reset 
     .FileName "*parabolicwork.sab" 
     .SubProjectName3D "..\..\Model\DS\Block\0\B2029407574\parabolicwork.cst" 
     .SubProjectScaleFactor "0.01" 
     .Version "14.0" 
     .PortnameMap "" 
     .AssemblyPartName "parabolicwork_1" 
     .ImportToActiveCoordinateSystem "True" 
     .Curves "True" 
     .Wires "True" 
     .SolidWiresAsSolids "False" 
     .ImportSources "True" 
     .Set "ImportSensitivityInformation", "True" 
     .Read 
End With


EndSubProject 


'@ transform local coordinates

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
TransformCurrentWCS "parabolicwork_1", True

'@ define frequency range

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
Solver.FrequencyRange "9", "10"

'@ define background

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With Background 
     .Reset 
     .XminSpace "0.0" 
     .XmaxSpace "0.0" 
     .YminSpace "0.0" 
     .YmaxSpace "0.0" 
     .ZminSpace "0.0" 
     .ZmaxSpace "0.0" 
     .ApplyInAllDirections "False" 
End With 

With Material 
     .Reset 
     .Rho "0"
     .ThermalType "Normal"
     .ThermalConductivity "0"
     .SpecificHeat "0", "J/K/kg"
     .DynamicViscosity "0"
     .UseEmissivity "True"
     .Emissivity "0"
     .MetabolicRate "0.0"
     .VoxelConvection "0.0"
     .BloodFlow "0"
     .Absorptance "0"
     .MechanicsType "Unused"
     .IntrinsicCarrierDensity "0"
     .FrqType "all"
     .Type "Normal"
     .MaterialUnit "Frequency", "Hz"
     .MaterialUnit "Geometry", "m"
     .MaterialUnit "Time", "s"
     .MaterialUnit "Temperature", "K"
     .Epsilon "1.0"
     .Mu "1.0"
     .Sigma "0"
     .TanD "0.0"
     .TanDFreq "0.0"
     .TanDGiven "False"
     .TanDModel "ConstSigma"
     .SetConstTanDStrategyEps "AutomaticOrder"
     .ConstTanDModelOrderEps "3"
     .DjordjevicSarkarUpperFreqEps "0"
     .SetElParametricConductivity "False"
     .ReferenceCoordSystem "Global"
     .CoordSystemType "Cartesian"
     .SigmaM "0"
     .TanDM "0.0"
     .TanDMFreq "0.0"
     .TanDMGiven "False"
     .TanDMModel "ConstSigma"
     .SetConstTanDStrategyMu "AutomaticOrder"
     .ConstTanDModelOrderMu "3"
     .DjordjevicSarkarUpperFreqMu "0"
     .SetMagParametricConductivity "False"
     .DispModelEps  "None"
     .DispModelMu "None"
     .DispersiveFittingSchemeEps "Nth Order"
     .MaximalOrderNthModelFitEps "10"
     .ErrorLimitNthModelFitEps "0.1"
     .UseOnlyDataInSimFreqRangeNthModelEps "False"
     .DispersiveFittingSchemeMu "Nth Order"
     .MaximalOrderNthModelFitMu "10"
     .ErrorLimitNthModelFitMu "0.1"
     .UseOnlyDataInSimFreqRangeNthModelMu "False"
     .UseGeneralDispersionEps "False"
     .UseGeneralDispersionMu "False"
     .NLAnisotropy "False"
     .NLAStackingFactor "1"
     .NLADirectionX "1"
     .NLADirectionY "0"
     .NLADirectionZ "0"
     .Colour "0.6", "0.6", "0.6" 
     .Wireframe "False" 
     .Reflection "False" 
     .Allowoutline "True" 
     .Transparentoutline "False" 
     .Transparency "0" 
     .ChangeBackgroundMaterial
End With

'@ define boundaries

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With Boundary 
     .Xmin "expanded open"
     .Xmax "expanded open"
     .Ymin "expanded open"
     .Ymax "expanded open"
     .Zmin "expanded open"
     .Zmax "expanded open"
     .Xsymmetry "none"
     .Ysymmetry "none"
     .Zsymmetry "none"
     .ApplyInAllDirections "False"
     .OpenAddSpaceFactor "0.5"
     .ThermalBoundary "Xmin", "isothermal"
     .ThermalBoundary "Xmax", "isothermal"
     .ThermalBoundary "Ymin", "isothermal"
     .ThermalBoundary "Ymax", "isothermal"
     .ThermalBoundary "Zmin", "isothermal"
     .ThermalBoundary "Zmax", "isothermal"
     .ThermalSymmetry "X", "none"
     .ThermalSymmetry "Y", "none"
     .ThermalSymmetry "Z", "none"
     .ResetThermalBoundaryValues
     .WallFlow "Xmin", "NoSlip"
     .EnableThermalRadiation "Xmin", "True"
     .WallFlow "Xmax", "NoSlip"
     .EnableThermalRadiation "Xmax", "True"
     .WallFlow "Ymin", "NoSlip"
     .EnableThermalRadiation "Ymin", "True"
     .WallFlow "Ymax", "NoSlip"
     .EnableThermalRadiation "Ymax", "True"
     .WallFlow "Zmin", "NoSlip"
     .EnableThermalRadiation "Zmin", "True"
     .WallFlow "Zmax", "NoSlip"
     .EnableThermalRadiation "Zmax", "True"
End With

'@ use template: Antenna - Reflector_1.cfg

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
'set the units
With Units
    .SetUnit "Length", "cm"
    .SetUnit "Frequency", "GHz"
    .SetUnit "Voltage", "V"
    .SetUnit "Resistance", "Ohm"
    .SetUnit "Inductance", "nH"
    .SetUnit "Temperature",  "degC"
    .SetUnit "Time", "ns"
    .SetUnit "Current", "A"
    .SetUnit "Conductance", "S"
    .SetUnit "Capacitance", "pF"
End With

ThermalSolver.AmbientTemperature "0"

'----------------------------------------------------------------------------

'set the frequency range
Solver.FrequencyRange "9", "10"

'----------------------------------------------------------------------------

Plot.DrawBox True

With Background
     .Type "Normal"
     .Epsilon "1.0"
     .Mu "1.0"
     .XminSpace "0.0"
     .XmaxSpace "0.0"
     .YminSpace "0.0"
     .YmaxSpace "0.0"
     .ZminSpace "0.0"
     .ZmaxSpace "0.0"
End With

With Boundary
     .Xmin "expanded open"
     .Xmax "expanded open"
     .Ymin "expanded open"
     .Ymax "expanded open"
     .Zmin "expanded open"
     .Zmax "expanded open"
     .Xsymmetry "none"
     .Ysymmetry "none"
     .Zsymmetry "none"
End With

' switch on FD-TET setting for accurate farfields

FDSolver.ExtrudeOpenBC "True"

Mesh.FPBAAvoidNonRegUnite "True"
Mesh.ConsiderSpaceForLowerMeshLimit "False"
Mesh.MinimumStepNumber "5"

With MeshSettings
     .SetMeshType "Hex"
     .Set "RatioLimitGeometry", "20"
End With

With MeshSettings
     .SetMeshType "HexTLM"
     .Set "RatioLimitGeometry", "20"
End With

PostProcess1D.ActivateOperation "vswr", "true"
PostProcess1D.ActivateOperation "yz-matrices", "true"

With FarfieldPlot
	.ClearCuts ' lateral=phi, polar=theta
	.AddCut "lateral", "0", "1"
	.AddCut "lateral", "90", "1"
	.AddCut "polar", "90", "1"
End With

'----------------------------------------------------------------------------

Dim sDefineAt As String
sDefineAt = "10"
Dim sDefineAtName As String
sDefineAtName = "10"
Dim sDefineAtToken As String
sDefineAtToken = "f="
Dim aFreq() As String
aFreq = Split(sDefineAt, ";")
Dim aNames() As String
aNames = Split(sDefineAtName, ";")

Dim nIndex As Integer
For nIndex = LBound(aFreq) To UBound(aFreq)

Dim zz_val As String
zz_val = aFreq (nIndex)
Dim zz_name As String
zz_name = sDefineAtToken & aNames (nIndex)

' Define E-Field Monitors
With Monitor
    .Reset
    .Name "e-field ("& zz_name &")"
    .Dimension "Volume"
    .Domain "Frequency"
    .FieldType "Efield"
    .MonitorValue  zz_val
    .Create
End With

' Define Farfield Monitors
With Monitor
    .Reset
    .Name "farfield ("& zz_name &")"
    .Domain "Frequency"
    .FieldType "Farfield"
    .MonitorValue  zz_val
    .ExportFarfieldSource "False"
    .Create
End With

Next

'----------------------------------------------------------------------------

With MeshSettings
     .SetMeshType "Srf"
     .Set "Version", 1%
End With

With Mesh
     .MeshType "Surface"
End With

'set the solver type
ChangeSolverType("HF IntegralEq")

'----------------------------------------------------------------------------

'@ define frequency range

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
Solver.FrequencyRange "9", "10"

'@ define background

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With Background 
     .Reset 
     .XminSpace "0.0" 
     .XmaxSpace "0.0" 
     .YminSpace "0.0" 
     .YmaxSpace "0.0" 
     .ZminSpace "0.0" 
     .ZmaxSpace "0.0" 
     .ApplyInAllDirections "False" 
End With 

With Material 
     .Reset 
     .Rho "0"
     .ThermalType "Normal"
     .ThermalConductivity "0"
     .SpecificHeat "0", "J/K/kg"
     .DynamicViscosity "0"
     .UseEmissivity "True"
     .Emissivity "0"
     .MetabolicRate "0.0"
     .VoxelConvection "0.0"
     .BloodFlow "0"
     .Absorptance "0"
     .MechanicsType "Unused"
     .IntrinsicCarrierDensity "0"
     .FrqType "all"
     .Type "Normal"
     .MaterialUnit "Frequency", "Hz"
     .MaterialUnit "Geometry", "m"
     .MaterialUnit "Time", "s"
     .MaterialUnit "Temperature", "K"
     .Epsilon "1.0"
     .Mu "1.0"
     .Sigma "0"
     .TanD "0.0"
     .TanDFreq "0.0"
     .TanDGiven "False"
     .TanDModel "ConstSigma"
     .SetConstTanDStrategyEps "AutomaticOrder"
     .ConstTanDModelOrderEps "3"
     .DjordjevicSarkarUpperFreqEps "0"
     .SetElParametricConductivity "False"
     .ReferenceCoordSystem "Global"
     .CoordSystemType "Cartesian"
     .SigmaM "0"
     .TanDM "0.0"
     .TanDMFreq "0.0"
     .TanDMGiven "False"
     .TanDMModel "ConstSigma"
     .SetConstTanDStrategyMu "AutomaticOrder"
     .ConstTanDModelOrderMu "3"
     .DjordjevicSarkarUpperFreqMu "0"
     .SetMagParametricConductivity "False"
     .DispModelEps  "None"
     .DispModelMu "None"
     .DispersiveFittingSchemeEps "Nth Order"
     .MaximalOrderNthModelFitEps "10"
     .ErrorLimitNthModelFitEps "0.1"
     .UseOnlyDataInSimFreqRangeNthModelEps "False"
     .DispersiveFittingSchemeMu "Nth Order"
     .MaximalOrderNthModelFitMu "10"
     .ErrorLimitNthModelFitMu "0.1"
     .UseOnlyDataInSimFreqRangeNthModelMu "False"
     .UseGeneralDispersionEps "False"
     .UseGeneralDispersionMu "False"
     .NLAnisotropy "False"
     .NLAStackingFactor "1"
     .NLADirectionX "1"
     .NLADirectionY "0"
     .NLADirectionZ "0"
     .Colour "0.6", "0.6", "0.6" 
     .Wireframe "False" 
     .Reflection "False" 
     .Allowoutline "True" 
     .Transparentoutline "False" 
     .Transparency "0" 
     .ChangeBackgroundMaterial
End With

'@ define boundaries

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With Boundary 
     .Xmin "expanded open"
     .Xmax "expanded open"
     .Ymin "expanded open"
     .Ymax "expanded open"
     .Zmin "expanded open"
     .Zmax "expanded open"
     .Xsymmetry "none"
     .Ysymmetry "none"
     .Zsymmetry "none"
     .ApplyInAllDirections "False"
     .OpenAddSpaceFactor "0.5"
     .ThermalBoundary "Xmin", "isothermal"
     .ThermalBoundary "Xmax", "isothermal"
     .ThermalBoundary "Ymin", "isothermal"
     .ThermalBoundary "Ymax", "isothermal"
     .ThermalBoundary "Zmin", "isothermal"
     .ThermalBoundary "Zmax", "isothermal"
     .ThermalSymmetry "X", "none"
     .ThermalSymmetry "Y", "none"
     .ThermalSymmetry "Z", "none"
     .ResetThermalBoundaryValues
     .WallFlow "Xmin", "NoSlip"
     .EnableThermalRadiation "Xmin", "True"
     .WallFlow "Xmax", "NoSlip"
     .EnableThermalRadiation "Xmax", "True"
     .WallFlow "Ymin", "NoSlip"
     .EnableThermalRadiation "Ymin", "True"
     .WallFlow "Ymax", "NoSlip"
     .EnableThermalRadiation "Ymax", "True"
     .WallFlow "Zmin", "NoSlip"
     .EnableThermalRadiation "Zmin", "True"
     .WallFlow "Zmax", "NoSlip"
     .EnableThermalRadiation "Zmax", "True"
End With

'@ define units

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With Units 
     .SetUnit "Length", "cm"
     .SetUnit "Temperature", "degC"
     .SetUnit "Voltage", "V"
     .SetUnit "Current", "A"
     .SetUnit "Resistance", "Ohm"
     .SetUnit "Conductance", "S"
     .SetUnit "Capacitance", "pF"
     .SetUnit "Inductance", "nH"
     .SetUnit "Frequency", "GHz"
     .SetUnit "Time", "ns"
     .SetResultUnit "frequency", "frequency", "" 
End With

'@ activate global coordinates

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
WCS.ActivateWCS "global"

'@ change problem type

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
ChangeProblemType "High Frequency"

'@ set solver type

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
SetSolverType "HF_INTEGRALEQUATION" 
ChangeSolverType "HF IntegralEq"

'@ HSDD: prepare domain

'[VERSION]2024.0|33.0.1|20230901[/VERSION]


Solver.FrequencyRange "9", "10"

With Boundary 
	.Xmin "expanded open" 
	.Xmax "expanded open" 
	.Ymin "expanded open" 
	.Ymax "expanded open" 
	.Zmin "expanded open" 
	.Zmax "expanded open" 
	.Xsymmetry "none" 
	.Ysymmetry "none" 
	.Zsymmetry "none" 
	.ApplyInAllDirections "True" 
	.MinimumDistanceReferenceFrequencyType "User" 
	.FrequencyForMinimumDistance "10" 
End With

With Background 
	.ResetBackground 
	.Type "Normal" 
	.XminSpace "0.0" 
	.XmaxSpace "0.0" 
	.YminSpace "0.0" 
	.YmaxSpace "0.0" 
	.ZminSpace "0.0" 
	.ZmaxSpace "0.0" 
	.ApplyInAllDirections "True" 
End With

'solver specific settings

With FDSolver 
	.ResetSampleIntervals ("all") 
	.AddInactiveSampleInterval "", "", "1", "Automatic", "False" 
	.AddMonitorSamples(true) 
	.SetMethod "Surface", "General purpose" 
	.ModesOnly "False" 
End With
With Mesh 
	.MeshType "Surface" 
	.SetCreator "High Frequency" 
End With
With MeshSettings 
	.SetMeshType "Srf" 
	.Set "Version", 1% 
End With
With IESolver 
	.UseFastFrequencySweep "False" 
	.PreconditionerType "Type 3" 
	.CalculateSParaforFieldsources "False" 
End With
DeleteGlobalDataValue("StepsPerLambdaForFieldMonitor")
With FDSolver 
	.UseEnhancedNFSImprint "True" 
End With
'Use meshed field source monitor
StoreGlobalDataValue("NewNFS_Monitor", "1")
DeleteGlobalDataValue("StepsPerLambdaForNFS")
'Use 2nd order for field source monitor
StoreGlobalDataValue("2ndOrderNFS_Monitor", "1")


Solver.AttachFFMIndepdentDataToFSM True

With TimeSignal 
	.ExcitationSignalAsReference "default", "High Frequency" 
End With
'@ HSDD: define field source monitors

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
WCS.ActivateWCS "local" 
WCS.AlignWCSWithGlobalCoordinates 
TransformCurrentWCS "parabolicwork_1", "TRUE" 
TransformCurrentWCS "hornwork_1", "FALSE" 
	With FieldSource 
		.Reset 
		.Name "fs1_(Horn)1" 
		.FileName "..\Result\DS\T_1\Horn\fs_Reflector_[Horn]1.fsm" 
		.Id "0" 
		.ImportToActiveCoordinateSystem "True" 
		.ImportMethod "LinkOnly" 
		.Read 
	End With
WCS.ActivateWCS "global"


Dim P1x As Double
Dim P1y As Double
Dim P1z As Double
Dim P2x As Double
Dim P2y As Double
Dim P2z As Double
Dim dx As Double
Dim dy As Double
Dim dz As Double
Dim distancex As Double
Dim distancey As Double
Dim distancez As Double
Pick.ClearAllPicks
Pick.PickPointFromIdOn "fieldsource$fs1_(Horn)1", "EndPoint", "1" 
Pick.PickPointFromIdOn "fieldsource$fs1_(Horn)1", "EndPoint", "2" 
Pick.GetPickpointCoordinatesByIndex 0, P1x, P1y, P1z
Pick.GetPickpointCoordinatesByIndex 1, P2x, P2y, P2z
dx = P2x-P1x
dy = P2y-P1y
dz = P2z-P1z
distancex = Sqr(dx*dx+dy*dy+dz*dz)
Pick.ClearAllPicks
Pick.PickPointFromIdOn "fieldsource$fs1_(Horn)1", "EndPoint", "1" 
Pick.PickPointFromIdOn "fieldsource$fs1_(Horn)1", "EndPoint", "4" 
Pick.GetPickpointCoordinatesByIndex 0, P1x, P1y, P1z
Pick.GetPickpointCoordinatesByIndex 1, P2x, P2y, P2z
dx = P2x-P1x
dy = P2y-P1y
dz = P2z-P1z
distancey = Sqr(dx*dx+dy*dy+dz*dz)
Pick.ClearAllPicks
Pick.PickPointFromIdOn "fieldsource$fs1_(Horn)1", "EndPoint", "1" 
Pick.PickPointFromIdOn "fieldsource$fs1_(Horn)1", "EndPoint", "5" 
Pick.GetPickpointCoordinatesByIndex 0, P1x, P1y, P1z
Pick.GetPickpointCoordinatesByIndex 1, P2x, P2y, P2z
dx = P2x-P1x
dy = P2y-P1y
dz = P2z-P1z
distancez = Sqr(dx*dx+dy*dy+dz*dz)
Pick.ClearAllPicks
'Coordinate system at front bottom left corner
Pick.PickPointFromIdOn "fieldsource$fs1_(Horn)1", "EndPoint", "1" 
Pick.PickPointFromIdOn "fieldsource$fs1_(Horn)1", "EndPoint", "2" 
Pick.PickPointFromIdOn "fieldsource$fs1_(Horn)1", "EndPoint", "4" 
WCS.AlignWCSWithSelected "3Points"

With Monitor 
	.Reset 
	.Name "fs_Horn" 
	.FieldType "Fieldsource" 

	.Domain "Frequency"
	.SamplingStrategy "Frequencies" 
	.MonitorValueList "10"
	.InvertOrientation "True" 
	.UseSubvolume "True" 
	.Coordinates "Free" 
	.UseWCSForSubvolume "True"
	.SetSubvolume "0", distancex, 0, distancey, 0, distancez
	.SetSubvolumeInflateWithOffset "True" 
	.SetSubvolumeOffset "20", "20", "20", "20", "20", "20" 
	.SetSubvolumeOffsetType "FractionOfWavelength" 
	.Create 
End With 
WCS.ActivateWCS "global"

Pick.ClearAllPicks


'@ HSDD: finalize domain setup

'[VERSION]2024.0|33.0.1|20230901[/VERSION]

With FieldSource
	.Reset
	.Select "fs\d+" 
	.Set "LossyCompression", True 
	.Modify
End With 

'@ set solver type

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
SetSolverType "HF_INTEGRALEQUATION" 
ChangeSolverType "HF IntegralEq"

'@ HSDD core configuration

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With FDSolver 
  .ResetSampleIntervals ("all") 
  .AddInactiveSampleInterval "", "", "1", "Automatic", "False" 
  .AddMonitorSamples(true) 
End With
With IESolver 
  .UseFastFrequencySweep "False" 
End With


With FDSolver
	.AddSampleInterval "10", "10", "1", "Single", "False"
End With
With Solver 
.ResetExcitationModes 
.SimultaneousExcitation "True" 
.SetSimultaneousExcitAutoLabel "False" 
.SetSimultaneousExcitationLabelNoCheck "[Horn]1" 
.ExcitationFieldSource "fs1_(Horn)1", "1.0", "0.0", "default", "True" 
.StimulationPort "Selected" 
.ExportSimultaneousExcitation
End With

With FDSolver 
  .ResetExcitationList 
  .Stimulation "List", "List" 
  .AddToExcitationList "Simultaneous", "[Horn]1;" 
End With

'@ HSDD observer definition

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With Monitor 
	.Reset 
	.FieldType "Farfield" 
	.UseSubvolume "False"
	.Domain "Frequency"
	.Name "farfield (f=10)"
	.Frequency "10"
	.Create
End With

With FarfieldPlot
	.Origin "free"
	.Userorigin "0", "0", "0"
	.SetAxesType "user"
	.Phistart "1", "0", "0" ' x'-axis
	.Thetastart "0", "0", "1" ' z'-axis
	.StoreSettings
End With

'@ HSDD activate FD excitations

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With FDSolver 
.ResetExcitationList 
.Stimulation "List", "List" 
.AddToExcitationList "Simultaneous", "[Horn]1" 
End With

'@ farfield plot options

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With FarfieldPlot 
     .Plottype "Polar" 
     .Vary "angle1" 
     .Theta "0" 
     .Phi "0" 
     .Step "5" 
     .Step2 "5" 
     .SetLockSteps "True" 
     .SetPlotRangeOnly "False" 
     .SetThetaStart "0" 
     .SetThetaEnd "180" 
     .SetPhiStart "0" 
     .SetPhiEnd "360" 
     .SetTheta360 "False" 
     .SymmetricRange "False" 
     .SetTimeDomainFF "False" 
     .SetFrequency "-1" 
     .SetTime "0" 
     .SetColorByValue "True" 
     .DrawStepLines "False" 
     .DrawIsoLongitudeLatitudeLines "False" 
     .ShowStructure "False" 
     .ShowStructureProfile "False" 
     .SetStructureTransparent "False" 
     .SetFarfieldTransparent "False" 
     .AspectRatio "Free" 
     .ShowGridlines "True" 
     .InvertAxes "False", "False" 
     .SetSpecials "enablepolarextralines" 
     .SetPlotMode "Directivity" 
     .Distance "1" 
     .UseFarfieldApproximation "True" 
     .IncludeUnitCellSidewalls "True" 
     .SetScaleLinear "False" 
     .SetLogRange "40" 
     .SetLogNorm "0" 
     .DBUnit "0" 
     .SetMaxReferenceMode "abs" 
     .EnableFixPlotMaximum "False" 
     .SetFixPlotMaximumValue "1.0" 
     .SetInverseAxialRatio "False" 
     .SetAxesType "user" 
     .SetAntennaType "unknown" 
     .Phistart "1.000000e+00", "0.000000e+00", "0.000000e+00" 
     .Thetastart "0.000000e+00", "0.000000e+00", "1.000000e+00" 
     .PolarizationVector "0.000000e+00", "1.000000e+00", "0.000000e+00" 
     .SetCoordinateSystemType "spherical" 
     .SetAutomaticCoordinateSystem "True" 
     .SetPolarizationType "Linear" 
     .SlantAngle 0.000000e+00 
     .Origin "free" 
     .Userorigin "0.000000e+00", "0.000000e+00", "0.000000e+00" 
     .SetUserDecouplingPlane "False" 
     .UseDecouplingPlane "False" 
     .DecouplingPlaneAxis "X" 
     .DecouplingPlanePosition "0.000000e+00" 
     .LossyGround "False" 
     .GroundEpsilon "1" 
     .GroundKappa "0" 
     .EnablePhaseCenterCalculation "False" 
     .SetPhaseCenterAngularLimit "3.000000e+01" 
     .SetPhaseCenterComponent "boresight" 
     .SetPhaseCenterPlane "both" 
     .ShowPhaseCenter "True" 
     .ClearCuts 
     .AddCut "lateral", "0", "1"  
     .AddCut "lateral", "90", "1"  
     .AddCut "polar", "90", "1"  

     .StoreSettings
End With

