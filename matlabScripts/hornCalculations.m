% Standard Gain Horn Antenna Simulation at 10 GHz
% Model: 16240-20

% Define single simulation frequency
frequency = 10e9;   % Frequency in Hz (10 GHz)

% Define Horn Antenna dimensions directly
apertureWidth = 109.25e-3;   % Aperture Width (B) in meters
apertureHeight = 79.00e-3;   % Aperture Height (C) in meters
flareLength = 232.50e-3; % Flare Length (F) in meters
waveguideWidth = 22.86e-3;
waveguideHeight = 10.16e-3;
waveguideLength = 14.1e-3;

% Create Horn Antenna using simplified horn aperture parameters
AUT = horn(FlareWidth = apertureWidth, ...
                   FlareHeight = apertureHeight, ...
                   FlareLength = flareLength, ...
                   Width = waveguideWidth, ...
                   Height = waveguideHeight, ...
                   Length = waveguideLength, ...
                   FeedOffset = [-5e-3 0]);

% Plot the antenna structure
figure;
show(AUT);
title('Standard Gain Horn Antenna Model');
axis equal;

% Compute and normalize the azimuth gain pattern
azimuthGain = patternAzimuth(AUT, frequency, azimuthAngles, 'Elevation', 0);

% Find the index of the maximum value in the azimuth gain
[maxGain, maxIndexAzimuth] = max(azimuthGain);

% Shift the azimuth angles so that the max gain is at 0 degrees
azimuthGainShifted = circshift(azimuthGain, -maxIndexAzimuth + find(azimuthAngles == 0));

% Normalize the shifted gain pattern to the peak value
azimuthGainShiftedNormalized = azimuthGainShifted - max(azimuthGainShifted);

% Compute and normalize the elevation gain pattern
elevationGain = patternElevation(AUT, frequency, elevationAngles, 'Azimuth', 0);

% Find the index of the maximum value in the elevation gain
[maxGain, maxIndexElevation] = max(elevationGain);

% Shift the elevation angles so that the max gain is at 0 degrees
elevationGainShifted = circshift(elevationGain, -maxIndexElevation + find(elevationAngles == 0));

% Normalize the shifted gain pattern to the peak value
elevationGainShiftedNormalized = elevationGainShifted - max(elevationGainShifted);

% Plot both patterns on the same figure
figure;
hold on;
plot(azimuthAngles, azimuthGainShiftedNormalized, 'b-', 'LineWidth', 1.5);
plot(elevationAngles, elevationGainShiftedNormalized, 'r-', 'LineWidth', 1.5);
hold off;

% Customize the plot
grid on;
title('Normalized Far-field Gain Pattern');
xlabel('Angle (degrees)');
ylabel('Gain (dB)');
legend('Azimuth Gain (Elevation = 0°)', 'Elevation Gain (Azimuth = 0°)', 'Location', 'best');

% Analyze and visualize radiation pattern at 10 GHz
figure;
pattern(AUT, frequency);
title('3D Radiation Pattern at 10 GHz');

% Plot Gain in Azimuth at 10 GHz
figure;
patternAzimuth(AUT, frequency, 0, 'Elevation', 0);  % Elevation = 0 degrees
title('Azimuth Gain Pattern at 10 GHz (Elevation = 0°)');

% Plot Gain in Elevation at 10 GHz
figure;
patternElevation(AUT, frequency, 0, 'Azimuth', 0);  % Azimuth = 0 degrees
title('Elevation Gain Pattern at 10 GHz (Azimuth = 0°)');


