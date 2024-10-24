'# MWS Version: Version 2024.0 - Sep 01 2023 - ACIS 33.0.1 -

'# length = cm
'# frequency = GHz
'# time = ns
'# frequency range: fmin = 9 fmax = 10
'# created = '[VERSION]2024.0|33.0.1|20230901[/VERSION]


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

'@ execute macro: Construct\Parts\Reflector dish

'[VERSION]2024.0|33.0.1|20230901[/VERSION]
Dim cst_D As Double, cst_focus As Double, cst_th_true As Boolean, cst_th_dish As Double
	Dim scst_D As Double, scst_th_true As Boolean, scst_th_dish As Double
	Dim cst_result As Integer
	Dim scst_focus As String

cst_result = -1%

	If (cst_result =0) Then Exit All   ' if cancel/help is clicked, exit all

	'  define curve circle: Dummy:circle1
	Dim sCurveName1 As String

	On Error GoTo Curve_Exists1
		Curve.NewCurve "Dummy"
	Curve_Exists1:
	On Error GoTo 0

	'check to generate a fresh dummy curve!
sCurveName1 = "circle_1"


	'  define curve circle: Dummy:circle1
	With Circle
	     .Reset
	     .Name sCurveName1
	     .Curve "Dummy"
	     .Radius "D/2"
	     .Xcenter "0.0"
	     .Ycenter "0.0"
	     .Segments "0"
	     .Create
	End With


	'  transform curve: translate Dummy:circle1
	With Transform
	     .Reset
	     .Name "Dummy:circle_1"
	     .Vector "0", "0", "(D^2)/(16*focus)+th_true-th_true"
	     .UsePickedPoints "False"
	     .InvertPickedPoints "False"
	     .MultipleObjects "False"
	     .GroupObjects "False"
	     .Repetitions "1"
	     .MultipleSelection "False"
	     .TranslateCurve
	End With

	'  check if curve exists and assign a name
	On Error GoTo Curve_Exists
		Curve.NewCurve "2D-Analytical"
	Curve_Exists:
	On Error GoTo 0

	Dim sCurveName As String

sCurveName = "profile_1"

	'  define analytical curve
	With AnalyticalCurve
	     .Reset
	     .Name sCurveName
	     .Curve "2D-Analytical"
	     .LawX "t"
	     .LawY "t^2/(4*focus)"
	     .LawZ "0"
	     .ParameterRange "0", "D/2"
	     .Create
	End With

	SelectTreeItem("Curves\2D-Analytical\"+sCurveName)

	'  transform curve: rotate 2D-Analytical:spline_1
	With Transform
	     .Reset
	     .Name "2D-Analytical:" + sCurveName
	     .Origin "Free"
	     .Center "0", "0", "0"
	     .Angle "90", "0", "0"
	     .MultipleObjects "False"
	     .GroupObjects "False"
	     .Repetitions "1"
	     .MultipleSelection "False"
	     .RotateCurve
	End With

	'  new component: antenna
	On Error GoTo Component_exists
		Component.New "antenna"
	Component_exists:
	On Error GoTo 0

	'  define sweepprofile: antenna:dish

	With SweepCurve
	     .Reset
	     .Name "reflector"
	     .Component "antenna"
	     .Material "PEC"
	     .Twistangle "0.0"
	     .Taperangle "0.0"
	     .ProjectProfileToPathAdvanced "False"
	     .Path "Dummy:circle_1"
	     .Curve "2D-Analytical:profile_1"
	     .Create
	End With

	SelectTreeItem("Components")

	'  delete curves

	Curve.DeleteCurve "2D-Analytical"
	Curve.DeleteCurve "Dummy"

	' add thickness if neccessary
	If RestoreDoubleParameter("th_true") Then
		Pick.PickFaceFromId "antenna:reflector", "1"
		Solid.ThickenSheetAdvanced "antenna:reflector", "Outside", "th_dish", "True"

		Pick.PickEndpointFromId "antenna:reflector", "2"
		Pick.MovePoint "-1", "0.0", "0.0", "focus", "False"   'translate the picked points in the focus of the reflector
	Else
		Pick.PickEndpointFromId "antenna:reflector", "1"
		Pick.MovePoint "-1", "0.0", "0.0", "focus", "False"   'translate the picked points in the focus of the reflector
	End If

	' handle distortion info

