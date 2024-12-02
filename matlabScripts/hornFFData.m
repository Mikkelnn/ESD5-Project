% Standard Gain Horn Antenna Simulation at 10 GHz
% Model: 16240-20

c = 3e8;
f = 10e9;
lam = c/f;
r_nf = 0.3;
N_1 = 100;
[X, Y, Z] = sphere(N_1);
Points_nf = [X(:), Y(:), Z(:)].'*r_nf;
N = N_1 + 1;

r_ff = 1000000;%1000
Points_ff = [X(:), Y(:), Z(:)].'*r_ff;

%--[x,y,z] = sph2cart(azimuth,elevation,r) transforms corresponding elements of the spherical coordinate arrays azimuth, elevation, and r to Cartesian, or xyz, coordinates.
%[azimuth,elevation,r] = cart2sph(x,y,z) transforms corresponding elements of the Cartesian coordinate arrays x, y, and z to spherical coordinates azimuth, elevation, and r.

[az,el,r] = cart2sph(X,Y,Z); %  el(-90:90) x az(-pi:pi)
az_1d = az(2,:);
az_1d(1) = -pi;
el_1d = el(:,1);
az_L = length(az_1d);
el_L = length(el_1d);

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
[E_nf, H_nf] = EHfields(AUT, f, Points_nf);
[E_ff, H_ff] = EHfields(AUT, f, Points_ff);

% Debug: Ensure size of E_nf is 3 x N
disp("Size of E_nf:");
disp(size(E_nf));
disp(size(H_nf))

% Calculate Scalar E_nf Magnitude
%E_nf_magnitude = sqrt(sum(abs(E_nf).^2, 1)); % Sum across rows (1st dimension), results in a 1 x N array


% Convert Cartesian to Spherical Coordinates
[X_sph, Y_sph, Z_sph] = deal(Points_nf(1, :), Points_nf(2, :), Points_nf(3, :));
[azimuth, elevation, r] = cart2sph(X_sph, Y_sph, Z_sph); % azimuth, elevation, r
spherical_theta = azimuth;            % Azimuth angle θ (in radians)
spherical_phi = pi/2 - elevation;     % Elevation φ (convert from elevation to colatitude)

% Save Near-Field Data
fileID_nf = fopen('simData\E_nf_reflector_spherical.txt', 'w');
fprintf(fileID_nf, 'r(m), theta(rad), phi(rad), E(Re), E(Im), H(Re), H(Im)\n');
for idx = 1:size(E_nf, 2) % Iterate over columns (points in space)
    % Extract real and imaginary parts of E
    E_real = sqrt(real(E_nf(1,idx))^2+real(E_nf(2,idx))^2+real(E_nf(3,idx))^2);
    E_imag = sqrt(abs(imag(E_nf(1,idx)))^2+abs(imag(E_nf(2,idx)))^2+abs(imag(E_nf(3,idx)))^2);
    H_real = sqrt(real(H_nf(1,idx))^2+real(H_nf(2,idx))^2+real(H_nf(3,idx))^2);
    H_imag = sqrt(abs(imag(H_nf(1,idx)))^2+abs(imag(H_nf(2,idx)))^2+abs(imag(H_nf(3,idx)))^2);
    % Write data to file
    fprintf(fileID_nf, '%f, %f, %f, %f, %f, %f, %f\n', ...
        r(idx), spherical_theta(idx), spherical_phi(idx), E_real, E_imag, H_real, H_imag);
end
fclose(fileID_nf);

% Repeat for Far-Field Data (E_ff)
%E_ff_magnitude = sqrt(sum(abs(E_ff).^2, 1)); % Sum across rows (1st dimension), results in a 1 x N array

[X_ff, Y_ff, Z_ff] = deal(Points_ff(1, :), Points_ff(2, :), Points_ff(3, :));
[azimuth_ff, elevation_ff, r_ff] = cart2sph(X_ff, Y_ff, Z_ff); % azimuth, elevation, r
spherical_theta_ff = azimuth_ff;        % Azimuth angle θ (in radians)
spherical_phi_ff = pi/2 - elevation_ff; % Elevation φ (convert from elevation to colatitude)

fileID_ff = fopen('simData\E_ff_reflector_spherical.txt', 'w');
fprintf(fileID_ff, 'r(m), theta(rad), phi(rad), E(Re), E(Im), H(Re), H(Im\n');
for idx = 1:size(E_ff, 2) % Iterate over columns (points in space)
    % Extract real and imaginary parts of E
    E_real = sqrt(real(E_ff(1,idx))^2+real(E_ff(2,idx))^2+real(E_ff(3,idx))^2);
    E_imag = sqrt(abs(imag(E_ff(1,idx)))^2+abs(imag(E_ff(2,idx)))^2+abs(imag(E_ff(3,idx)))^2);
    H_real = sqrt(real(H_ff(1,idx))^2+real(H_ff(2,idx))^2+real(H_ff(3,idx))^2);
    H_imag = sqrt(abs(imag(H_ff(1,idx)))^2+abs(imag(H_ff(2,idx)))^2+abs(imag(H_ff(3,idx)))^2);
    % Write data to file
    fprintf(fileID_ff, '%f, %f, %f, %f, %f, %f, %f\n', ...
        r_ff(idx), spherical_theta_ff(idx), spherical_phi_ff(idx), E_real, E_imag, H_real, H_imag);
end
fclose(fileID_ff);
