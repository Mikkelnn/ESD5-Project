clc;
close all;
clear all;
c = 3e8;
f = 10e9;
lam = c/f;
r_nf = lam*10;
N_1 = 100;
[X, Y, Z] = sphere(N_1);
Points_nf = [X(:), Y(:), Z(:)].'*r_nf;
N = N_1 + 1;

r_ff = r_nf*10000;%1000
Points_ff = [X(:), Y(:), Z(:)].'*r_ff;

%--[x,y,z] = sph2cart(azimuth,elevation,r) transforms corresponding elements of the spherical coordinate arrays azimuth, elevation, and r to Cartesian, or xyz, coordinates.
%[azimuth,elevation,r] = cart2sph(x,y,z) transforms corresponding elements of the Cartesian coordinate arrays x, y, and z to spherical coordinates azimuth, elevation, and r.

[az,el,r] = cart2sph(X,Y,Z); %  el(-90:90) x az(-pi:pi)
az_1d = az(2,:);
az_1d(1) = -pi;
el_1d = el(:,1);
az_L = length(az_1d);
el_L = length(el_1d);

if 1
    ant = reflectorParabolic;
    design(reflectorParabolic,f);
    

    [bw, angles] = beamwidth(ant, f, 0, 1:1:360);

    figure(1);
    pattern(ant, f);

    figure(2);
    show(ant);

    V = pattern(ant,f,0,0:1:360);
    P = polarpattern(V);

    % figure(3);
    % EHfields(ant, f, Points);
    %
    % figure(4);
    % subplot(2,1,1);
    % EHfields(ant, f, Points, ViewField="H");
    % subplot(2,1,2);
    % EHfields(ant, f, Points, ViewField="E", ScaleFields=[2,0]);

  
    %plot3(Points(1,:), Points(2,:), Points(3,:), 'x');
    %x, y, z components of electrical field in the rectangular coordinate system or azimuth, elevation, radial components in the spherical coordinate system,
    [E_nf, H_nf] = EHfields(ant, f, Points_nf);
  
    [E_ff, H_ff] = EHfields(ant, f, Points_ff);
    save('simData\E_nf_reflector.mat','E_nf');
    save('simData\E_ff_reflector.mat','E_ff');
else
    load('simData\E_nf_reflector.mat');%,'E_nf'); 3 x 10201
    load('simData\E_ff_reflector.mat');%,'E_ff');
end

E_total_nf = sum(E_nf.*conj(E_nf),1);
E_total_nf = reshape(E_total_nf,[el_L,az_L]);
max_nf = 10*log10(max(abs(E_total_nf(:))));

E_total_ff = sum(E_ff.*conj(E_ff),1);
E_total_ff = reshape(E_total_ff,[el_L,az_L]);
max_ff = 10*log10(max(abs(E_total_ff(:))));

figure;
subplot(2,1,1);
surf(rad2deg(az_1d),rad2deg(el_1d),10*log10(abs(E_total_nf))-max_nf);
shading flat;
xlabel('az [deg]');
ylabel('el [deg]');
caxis([-30,0]);
colorbar;
view(0,90);
title(['NF-E-total-max: ',num2str(round(max_nf*10)/10),'dB']);
subplot(2,1,2);
surf(rad2deg(az_1d),rad2deg(el_1d),10*log10(abs(E_total_ff))-max_ff);
shading flat;
xlabel('az [deg]');
ylabel('el [deg]');
caxis([-30,0]);
colorbar;
view(0,90);
title(['FF-E-total-max: ',num2str(round(max_ff*10)/10),'dB']);

figure;
subplot(2,1,1);
plot(rad2deg(az_1d),10*log10(abs([E_total_nf(51:101,1); E_total_nf(100:-1:51,51)]))-max_nf,'b','linewidth',1.2);hold on;
plot(rad2deg(az_1d),10*log10(abs([E_total_ff(51:101,1); E_total_ff(100:-1:51,51)]))-max_ff,'r','linewidth',1.2);hold on;
xlabel('az [deg]');
subplot(2,1,2);
plot(rad2deg(el_1d),10*log10(abs([E_total_nf(51:101,26); E_total_nf(100:-1:51,51+25)]))-max_nf,'b','linewidth',1.2);hold on;
plot(rad2deg(el_1d),10*log10(abs([E_total_ff(51:101,26); E_total_ff(100:-1:51,51+25)]))-max_ff,'r','linewidth',1.2);hold on;
xlabel('el [deg]');
legend('NF','FF');





