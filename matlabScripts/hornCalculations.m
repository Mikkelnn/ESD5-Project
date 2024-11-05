% MATLAB Script to calculate far-field gain pattern of a rectangular horn antenna
% Frequency and aperture dimensions are adjustable

% Constants
c = 3e8;                % Speed of light (m/s)

% Parameters (adjust these as needed)
frequency = 10e9;       % Frequency in Hz (e.g., 10 GHz)
lambda = c / frequency; % Wavelength in meters
a = 270e-3;                % Aperture width (along y-axis) in meters (e.g., 0.1 m)
b = 146.2e-3;               % Aperture height (along z-axis) in meters (e.g., 0.08 m)
eta = 0.7;              % Antenna efficiency (0.5 to 0.8 typical)

% Derived parameters
A = a * b;              % Aperture area
D = (4 * pi * A) / lambda^2; % Directivity
G0 = eta * D;           % Maximum gain

% Define theta and phi (in radians)
theta = linspace(-pi/2, pi/2, 1000);  % Angle range for pattern
phi_e_plane = 0;        % Phi for E-plane (y-axis)
phi_h_plane = pi/2;     % Phi for H-plane (z-axis)

% Far-field gain pattern (in terms of theta)
% E-plane (phi = 0)
E_theta_e = cos(pi * a * sin(theta) / lambda) .* sinc(b * sin(theta) / lambda);
G_e_plane = G0 * abs(E_theta_e).^2;

% H-plane (phi = pi/2)
E_theta_h = sinc(a * sin(theta) / lambda) .* sinc(b * sin(theta) / lambda);
G_h_plane = G0 * abs(E_theta_h).^2;

% Convert gain to dB
G_e_plane_dB = 10 * log10(G_e_plane);
G_h_plane_dB = 10 * log10(G_h_plane);

% Plotting the results
figure;
subplot(2,1,1);
plot(theta * 180/pi, G_e_plane_dB, 'LineWidth', 2);
title('Far-Field Gain Pattern - E-plane');
xlabel('Theta (degrees)');
ylabel('Gain (dB)');
grid on;
xlim([-30 30]);

subplot(2,1,2);
plot(theta * 180/pi, G_h_plane_dB, 'LineWidth', 2);
title('Far-Field Gain Pattern - H-plane');
xlabel('Theta (degrees)');
ylabel('Gain (dB)');
grid on;
xlim([-30 30]);

% Display maximum gain
fprintf('Antenna Parameters:\n');
fprintf('Frequency: %.2f GHz\n', frequency/1e9);
fprintf('Aperture Dimensions: %.2f m x %.2f m\n', a, b);
fprintf('Max Gain (dB): %.2f dB\n', 10*log10(G0));
