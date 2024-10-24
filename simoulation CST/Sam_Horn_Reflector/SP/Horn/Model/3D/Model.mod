'# MWS Version: Version 2024.0 - Sep 01 2023 - ACIS 33.0.1 -

'# length = in
'# frequency = GHz
'# time = ns
'# frequency range: fmin = 8 fmax = 10
'# created = '[VERSION]2024.0|33.0.1|20230901[/VERSION]


'@ define units

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With Units 
     .SetUnit "Length", "in"
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
SetWCSFromReferenceBlockInAssembly "hornwork_1"

'@ transform local coordinates

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
TransformCurrentWCS "hornwork_1", False

'@ import external project: ..\..\Model\DS\Block\0\B777257269\hornwork.cst

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
StartSubProject 

With Material 
     .Reset 
     .Name "PEC" 
     .Folder "hornwork_1" 
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
     .FileName "*hornwork.sab" 
     .SubProjectName3D "..\..\Model\DS\Block\0\B777257269\hornwork.cst" 
     .SubProjectScaleFactor "0.0254" 
     .Version "14.0" 
     .PortnameMap "1>1" 
     .SubProjectLocalWCS "0", "0", "2.5", "0", "0", "1", "1", "0", "0" 
     .AssemblyPartName "hornwork_1" 
     .ImportToActiveCoordinateSystem "True" 
     .Curves "True" 
     .Wires "True" 
     .SolidWiresAsSolids "False" 
     .ImportSources "True" 
     .Set "ImportSensitivityInformation", "True" 
     .Read 
End With

With Port 
     .Reset 
     .PortNumber "1" 
     .Label ""
     .Folder "hornwork_1"
     .NumberOfModes "1"
     .AdjustPolarization "False"
     .PolarizationAngle "0"
     .ReferencePlaneDistance "0"
     .TextSize "50"
     .TextMaxLimit "0"
     .Coordinates "Free"
     .Orientation "zmin"
     .PortOnBound "False"
     .ClipPickedPortToBound "False"
     .Xrange "0", "1"
     .Yrange "0", "0.5"
     .Zrange "0", "0"
     .XrangeAdd "0.0", "0.0"
     .YrangeAdd "0.0", "0.0"
     .ZrangeAdd "0.0", "0.0"
     .SingleEnded "False"
     .WaveguideMonitor "False"
     .ReferenceWCS "0.5", "0.25", "0", "0", "0", "-1", "0", "1", "0"
     .CreateImported 
End With 

With Transform 
     .Reset 
     .Name "port1" 
     .UseGlobalCoordinates "True" 
     .Vector "-0.5", "-0.25", "-2.5" 
     .AdjustVectorToSubProjectScaleFactor 
     .Matrix "1", "0", "0", "0", "1", "0", "0", "0", "1" 
     .Transform "port", "matrix" 
     .Transform "port", "GlobalToLocal" 
     .UseGlobalCoordinates "False" 
     .Vector  "0", "0", "2.5" 
     .AdjustVectorToSubProjectScaleFactor 
     .Matrix  "1", "0", "0", "0", "1", "0", "0", "0", "1" 
     .Transform "port", "matrix" 
End With 


EndSubProject 


'@ transform local coordinates

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
TransformCurrentWCS "hornwork_1", True

'@ define frequency range

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
Solver.FrequencyRange "8", "12"

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
     .Xsymmetry "magnetic"
     .Ysymmetry "electric"
     .Zsymmetry "none"
     .ApplyInAllDirections "False"
     .OpenAddSpaceFactor "0.5"
     .ThermalBoundary "Xmin", "isothermal"
     .ThermalBoundary "Xmax", "isothermal"
     .ThermalBoundary "Ymin", "isothermal"
     .ThermalBoundary "Ymax", "isothermal"
     .ThermalBoundary "Zmin", "isothermal"
     .ThermalBoundary "Zmax", "isothermal"
     .ThermalSymmetry "X", "symmetric"
     .ThermalSymmetry "Y", "symmetric"
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

'@ use template: Antenna - Waveguide_1.cfg

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
'set the units
With Units
    .SetUnit "Length", "in"
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
Solver.FrequencyRange "8", "12"

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
     .SetMeshType "Hex"
     .Set "Version", 1%
End With

With Mesh
     .MeshType "PBA"
End With

'set the solver type
ChangeSolverType("HF Time Domain")

'----------------------------------------------------------------------------

'@ define time domain solver parameters

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
Mesh.SetCreator "High Frequency" 

With Solver 
     .Method "Hexahedral"
     .CalculationType "TD-S"
     .StimulationPort "All"
     .StimulationMode "All"
     .SteadyStateLimit "-40"
     .MeshAdaption "False"
     .AutoNormImpedance "False"
     .NormingImpedance "50"
     .CalculateModesOnly "False"
     .SParaSymmetry "False"
     .StoreTDResultsInCache  "False"
     .RunDiscretizerOnly "False"
     .FullDeembedding "False"
     .SuperimposePLWExcitation "False"
     .UseSensitivityAnalysis "False"
End With

'@ farfield plot options

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With FarfieldPlot 
     .Plottype "3D" 
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
     .SetFrequency "10" 
     .SetTime "0" 
     .SetColorByValue "True" 
     .DrawStepLines "False" 
     .DrawIsoLongitudeLatitudeLines "False" 
     .ShowStructure "True" 
     .ShowStructureProfile "True" 
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
     .Origin "bbox" 
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

     .StoreSettings
End With

'@ define frequency range

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
Solver.FrequencyRange "8", "12"

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
     .Xsymmetry "magnetic"
     .Ysymmetry "electric"
     .Zsymmetry "none"
     .ApplyInAllDirections "False"
     .OpenAddSpaceFactor "0.5"
     .ThermalBoundary "Xmin", "isothermal"
     .ThermalBoundary "Xmax", "isothermal"
     .ThermalBoundary "Ymin", "isothermal"
     .ThermalBoundary "Ymax", "isothermal"
     .ThermalBoundary "Zmin", "isothermal"
     .ThermalBoundary "Zmax", "isothermal"
     .ThermalSymmetry "X", "symmetric"
     .ThermalSymmetry "Y", "symmetric"
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
     .SetUnit "Length", "in"
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
SetSolverType "HF_TRANSIENT" 
ChangeSolverType "HF Time Domain" 
With Solver
     .Method "Hexahedral"
End With

'@ HSDD: prepare domain

'[VERSION]2024.0|33.0.1|20230901[/VERSION]


Solver.FrequencyRange "8", "10"

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

With Solver 
	.Method "Hexahedral" 
	.MeshAdaption "False" 
	.CalculateModesOnly "False" 
.SetModeFreqFactor "1.0"
.AdaptivePortMeshing "False"
End With
With Mesh 
	.MeshType "PBA" 
	.SetCreator "High Frequency" 
End With
With MeshSettings 
	.SetMeshType "Hex" 
	.Set "Version", 1% 
End With
With TimeSignal 
	.Reset 
	.Name "default" 
	.SignalType "Gaussian sine" 
	.ProblemType "High Frequency" 
	.Fmin "8" 
	.Fmax "10" 
	.Create 
End With

Solver.AttachFFMIndepdentDataToFSM True

With TimeSignal 
	.ExcitationSignalAsReference "default", "High Frequency" 
End With
'@ HSDD: define field source monitors

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
Solid.CalculateExactBoundingBox 
With Monitor 
	.Reset 
	.Name "fs_Reflector" 
	.FieldType "Fieldsource" 

	.Domain "Frequency"
	.SamplingStrategy "Frequencies" 
	.MonitorValueList "10"
	.InvertOrientation "False" 
	.UseSubvolume "True" 
	.Coordinates "StructureNoUpdate" 
	.SetSubvolumeOffset "20", "20", "20", "20", "20", "20" 
	.SetSubvolumeInflateWithOffset "True" 
	.SetSubvolumeOffsetType "FractionOfWavelength" 
	.Create 
End With
'@ HSDD: finalize domain setup

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
Dim varIsCurrentWCSAxisAligned As Boolean
WCS.ActivateWCS "local" 
WCS.AlignWCSWithGlobalCoordinates 
TransformCurrentWCS "hornwork_1", "TRUE" 
TransformCurrentWCS "parabolicwork_1", "FALSE" 
	With FieldSource 
		.Reset 
		.Name "fs1_(Horn)1" 
		.FileName "..\Result\DS\T_1\Reflector\fs_Horn_[Horn]1.fsm" 
		.Id "0" 
		.ImportToActiveCoordinateSystem "True" 
		.ImportMethod "LinkOnly" 
		.Read 
	End With
Group.Add "HS_Local", "mesh"
With MeshSettings
     With .ItemMeshSettings ("group$HS_Local")
          .SetMeshType "Hex"
          .Set "EdgeRefinement", "1"
          .Set "Extend", "1", "1", "1"
          .Set "Fixpoints", 1
          .Set "MeshType", "Default"
          .Set "NumSteps", "1", "1", "1"
          .Set "Priority", "0"
          .Set "RefinementPolicy", "RATIO"
          .Set "SnappingIntervals", 0, 0, 0
          .Set "SnappingPriority", 0
          .Set "SnapTo", "1", "1", "1"
          .Set "Step", "0", "0", "0"
          .Set "StepRatio", "4", "4", "4"
          .Set "StepRefinementCollectPolicy", "REFINE_BOUNDARIES"
          .Set "StepRefinementExtentPolicy", "EXTENT_NUM_CELLS"
          .Set "UseDielectrics", 1
          .Set "UseEdgeRefinement", 0
          .Set "UseForRefinement", 1
          .Set "UseForSnapping", 1
          .Set "UseSameExtendXYZ", 1
          .Set "UseSameStepWidthXYZ", 1
          .Set "UseSnappingPriority", 0
          .Set "UseStepAndExtend", 1
          .Set "UseVolumeRefinement", 0
          .Set "VolumeRefinement", "1"
     End With
End With
varIsCurrentWCSAxisAligned = WCS.IsCurrentWCSAxisAligned()
if (varIsCurrentWCSAxisAligned) Then
Group.AddItem "fieldsource$fs1_(Horn)1", "HS_Local"
End If
WCS.ActivateWCS "global"

Component.HideAllFieldSources

With FieldSource
	.Reset
	.Select "fs\d+" 
	.Set "LossyCompression", True 
	.Modify
End With 

'@ set reference block coordinate system in assembly: hornwork_1

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
SetWCSFromReferenceBlockInAssembly "hornwork_1"

'@ start importing platform

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With Material 
	.Reset 
	.Name "UnconsideredMaterial" 
	.Folder "HSDD" 
	.Type "Normal" 
	.Epsilon "1" 
	.Mu "1" 
	.Colour "0", "0.501961", "0.25098" 
	.Wireframe "True"  
	.Reflection "False" 
	.Allowoutline "False" 
	.Transparentoutline "True" 
	.Transparency "100" 
	.Create 
End With
Dim preservedWCSState As String 
Dim minX As Double 
Dim maxX As Double 
Dim minY As Double 
Dim maxY As Double 
Dim minZ As Double 
Dim maxZ As Double 
preservedWCSState = WCS.IsWCSActive 
WCS.ActivateWCS "global" 
If IsCurrentlyFastLoading Then
	minX = 0
	maxX = 1
	minY = 0
	maxY = 1
	minZ = 0
	maxZ = 1
Else
	Boundary.GetCalculationBox minX, maxX, minY, maxY, minZ, maxZ
End If
With Brick 
	.Reset 
	.Name "EnsuresCalculationDomainSize" 
	.Component "HSDD" 
	.Material "HSDD/UnconsideredMaterial" 
	.Xrange minX, maxX 
	.Yrange minY, maxY 
	.Zrange minZ, maxZ 
	.Create 
End With 
WCS.ActivateWCS preservedWCSState
Component.HideComponent "HSDD"
Group.AddItem "solid$HSDD:EnsuresCalculationDomainSize", "Excluded from Simulation"
With Boundary 
	.Xmin "open" 
	.Xmax "open" 
	.Ymin "open" 
	.Ymax "open" 
	.Zmin "open" 
	.Zmax "open" 
	.Xsymmetry "none" 
	.Ysymmetry "none" 
	.Zsymmetry "none" 
	.ApplyInAllDirections "True" 
	.MinimumDistanceReferenceFrequencyType "User" 
End With
Assembly.StartImportingPlatform 
Assembly.SetSourceOffset "0", "0", "0", "0", "0", "0" 

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

'@ end importing platform

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
Assembly.SetResolveGeometryConflicts 
Assembly.EndImportingPlatform 

'@ HSDD: activate global coordinates

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
WCS.ActivateWCS "global"

'@ set solver type

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
SetSolverType "HF_TRANSIENT" 
ChangeSolverType "HF Time Domain" 
With Solver
     .Method "Hexahedral"
End With

'@ HSDD core configuration

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With Solver 
.ResetExcitationModes 
.SimultaneousExcitation "True" 
.SetSimultaneousExcitAutoLabel "False" 
.SetSimultaneousExcitationLabelNoCheck "[Horn]1" 
.ExcitationPortMode "1", "1", "1.0", "0.0", "default", "True" 
.StimulationPort "Selected" 
.ExportSimultaneousExcitation
End With

With Solver
	.ResetExcitationModes
	.SParameterPortExcitation "True"
	.SimultaneousExcitation "False"
	.DefineExcitationSettings "simultaneous", "[Horn]1", "1", "1.0", "0.0", "default", "default", "default", "default", "True"
End With

'@ set PBA version

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
Discretizer.PBAVersion "2023090124"

'@ HSDD core configuration

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With Solver 
.ResetExcitationModes 
.SimultaneousExcitation "True" 
.SetSimultaneousExcitAutoLabel "False" 
.SetSimultaneousExcitationLabelNoCheck "[Horn]1" 
.ExcitationFieldSource "fs1_(Horn)1", "1.0", "0.0", "default", "True" 
.ExcitationPortMode "1", "1", "1.0", "0.0", "default", "True" 
.StimulationPort "Selected" 
.ExportSimultaneousExcitation
End With

With Solver
	.ResetExcitationModes
	.SParameterPortExcitation "True"
	.SimultaneousExcitation "False"
	.DefineExcitationSettings "simultaneous", "[Horn]1", "1", "1.0", "0.0", "default", "default", "default", "default", "True"
End With

'@ HSDD observer definition

'[VERSION]2024.0|33.0.1|20230901[/VERSION]

'@ HSDD activate FD excitations

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
With FDSolver 
.ResetExcitationList 
.Stimulation "List", "List" 
.AddToExcitationList "Simultaneous", "[Horn]1" 
End With

