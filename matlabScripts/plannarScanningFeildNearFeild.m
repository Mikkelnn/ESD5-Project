% MATLAB code to simulate the planar scanning field in the near field
% of an isotropic antenna and prepare the data for far-field computation

clc;
clear;
close all;

% Constants
c = 3e8;          % Speed of light (m/s)
f = 1e9;          % Frequency (Hz)
lambda = c / f;   % Wavelength (m)
k = 2 * pi / lambda; % Wavenumber

% Antenna parameters
antenna_position = [0, 0, 0]; % Isotropic antenna at the origin
E0 = 1;           % Electric field strength at the antenna (arbitrary units)

% Define expanded scanning plane dimensions (centered)
x_scan_range = linspace(-3, 3, 300);  % Larger scanning range from -1 to 1 meters with 200 points
y_scan_range = linspace(-3, 3, 300);  
z_scan = 3;    % Distance from antenna to scanning plane (meters)

[X, Y] = meshgrid(x_scan_range, y_scan_range);  % Create a grid for the scanning plane
Z = z_scan * ones(size(X));                    % Z-coordinate of the scanning plane

% Compute the distance from the antenna to each point on the scanning plane
R = sqrt((X - antenna_position(1)).^2 + ...
         (Y - antenna_position(2)).^2 + ...
         (Z - antenna_position(3)).^2);

% Compute the electric field at each point on the scanning plane (near-field approximation)
E_field = E0 * exp(-1j * k * R) ./ R;

% Plot the magnitude of the electric field in the scanning plane
figure;
imagesc(x_scan_range, y_scan_range, abs(E_field));
colorbar;
title('Near-Field Magnitude of an Isotropic Antenna');
xlabel('X (m)');
ylabel('Y (m)');
axis equal;
axis tight;

% Plot the phase of the electric field in the scanning plane
figure;
imagesc(x_scan_range, y_scan_range, angle(E_field));
colorbar;
title('Phase of Electric Field in Near Field');
xlabel('X (m)');
ylabel('Y (m)');
axis equal;
axis tight;

% Vectorize the field data for far-field computation
E_vector = reshape(E_field, [], 1);  % Flatten the 2D electric field into a 1D vector
X_vector = reshape(X, [], 1);        % Flatten the X-coordinates
Y_vector = reshape(Y, [], 1);        % Flatten the Y-coordinates
Z_vector = z_scan * ones(size(X_vector));  % Z-coordinates (constant for all points)

% Combine into a single matrix (position + field values)
field_data = [X_vector, Y_vector, Z_vector, real(E_vector), imag(E_vector)];

% Save the field data for further use (Far-Field Computation)
% You can export it to a file if needed for external use:
% save('near_field_data.mat', 'field_data');

% Display field_data format
disp('Field Data Format: [X, Y, Z, Real(E), Imag(E)]');
disp(field_data);
