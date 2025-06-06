import os
import json
import logging

import numpy as np
import pandas as pd
import plotnine as p9
import matplotlib.pyplot as plt

from glob import glob
from copy import deepcopy
from tqdm.auto import tqdm
from matplotlib.collections import PolyCollection
from typing import Tuple, Dict, Union, List, Optional
from scipy.interpolate import InterpolatedUnivariateSpline

plt.style.use('seaborn-v0_8-whitegrid')

# Lifetime: Intensity collected for 4 different wavelength ranges for 64 ns with a sampling rate of 1 ns
N_LIFETIME_TIMESTEPS = 64
WAVELENGTH_LIFETIME_RANGES = ['350-400 nm', '420-460 nm', '511-572 nm', '672-800 nm']
# Scattering: According to description, 24 detectors in angles ranging from 45 to 135ª
N_SCATTERING_PIXELS = 24
SCATTERING_ANGLES = np.linspace(45, 135, N_SCATTERING_PIXELS)
IDX_TO_ANGLE = dict((i, angle) for i, angle in enumerate(SCATTERING_ANGLES))
# Spectrometer: 32 detectors at wavelengths ranging from 290 to 600 nm with 8 acquisitions separated by 500 ns
SPECTRAL_WAVELENGTHS = np.linspace(290, 600, 32)
# Fix because of design error notified by Denis...
spectrum = np.array(range(32))
spectrum_corrected = np.zeros(spectrum.shape)
spectrum_corrected[::2] = spectrum[1::2]
spectrum_corrected[1::2] = spectrum[::2]
spectrum_corrected = [int(i) for i in spectrum_corrected]
CORRECTED_SPECTRAL_WAVELENGTHS = SPECTRAL_WAVELENGTHS[spectrum_corrected]
SPECTRAL_TIME = np.arange(0, 4, .5)
FLUORESCENCE_THRESHOLD = 600


class ParticleData(object):
    def __init__(self, data: Dict[str, Union[Dict[str, List[float]], List[float]]]):
        """
        Container for all data and computed variables relevant for a single aerosol particle

        Parameters
        ----------
        data
            Dictionary read from the json file outputted by `conversion.py` containing the data from a single particle.
            A json with n particles contains n `data` dictionaries. Accessed such that `json['Data'][particle_n]`
        """
        self.raw_lifetime = data['Lifetime']
        self.raw_scattering = np.array(data['Scattering']['Image'])
        self.raw_spectral = data['Spectrometer']
        self.scattering_acquisitions = int(self.raw_scattering.shape[0] / N_SCATTERING_PIXELS)
        self.size = data['Size']
        # Adding AirLab coords until proper GPS data is fetched
        self.latitude = 41.38441
        self.longitude = 2.186476
        self.timestamp = data['Timestamp']
        self.max_wavelength, self.max_time, self.max_intensity = self.max_intensity_properties.values[0]
        self.fluorescent = self.max_intensity > FLUORESCENCE_THRESHOLD

    @property
    def scattering(self):
        return (pd.DataFrame(self.raw_scattering.reshape(-1, N_SCATTERING_PIXELS))
                .assign(time=lambda dd: np.arange(0, dd.shape[0] / 2, .5))
                .melt('time', value_name='intensity')
                .assign(angle=lambda dd: dd.variable.replace(IDX_TO_ANGLE))
                .drop(columns='variable'))

    @property
    def scattering_matrix(self):
        return self.scattering.round(1).pivot(index='time', columns='angle').intensity

    @property
    def lifetime(self):
        return pd.DataFrame(
            {'intensity': self.raw_lifetime,
             'time': np.tile(np.arange(N_LIFETIME_TIMESTEPS), len(WAVELENGTH_LIFETIME_RANGES)),
             'wavelength_range': np.repeat(WAVELENGTH_LIFETIME_RANGES, N_LIFETIME_TIMESTEPS)
             })

    @property
    def lifetime_matrix(self):
        return self.lifetime.pivot('time', 'wavelength_range')

    @property
    def spectral_data(self):
        return (pd.DataFrame(np.array(self.raw_spectral).reshape(len(SPECTRAL_WAVELENGTHS), len(SPECTRAL_TIME)))
                .rename(columns={i: wl for i, wl in enumerate(SPECTRAL_TIME)})
                .assign(wavelength=CORRECTED_SPECTRAL_WAVELENGTHS)
                .melt('wavelength', var_name='time', value_name='intensity'))

    @property
    def spectrum_time_matrix(self):
        return self.spectral_data.pivot(index='wavelength', columns='time')

    @property
    def max_intensity_properties(self):
        max_intensity_props = self.spectral_data.loc[lambda dd: dd.intensity == dd.intensity.max()]
        if max_intensity_props.shape[0] > 1:
            logging.debug('Particle has multiple max intensities at different wavelengths/times. Keeping only first.')
        return max_intensity_props.iloc[0].to_frame().T.astype(float)

    def plot_lifetime(self, dim: str = '2d', figsize: Tuple[int, int] = (9, 6), dpi: int = 120):
        figsize = figsize if figsize is not None else (9, 6)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        if dim.lower() == '2d':
            ax = plt.gca()
            self.lifetime_matrix.where(lambda x: x > 50, other=0).plot(ax=ax)
            ax.set_xlabel('Time [ns]')
            ax.set_ylabel('Intensity')
            ax.legend(title='Wavelength Range', labels=WAVELENGTH_LIFETIME_RANGES)

        elif dim.lower() == '3d':
            ax = fig.gca(projection='3d')
            sample = []
            # We will pad the ys and xs so that the polygon is filled starting at the plane y=0
            xs = np.concatenate([[-1], np.arange(65)])
            zs = np.arange(4)
            for _, df in self.lifetime.groupby('wavelength_range'):
                ys = np.pad(df.intensity.where(lambda x: x > 100, other=0), 1)
                sample.append((list(zip(xs, ys))))
            poly = PolyCollection(sample, facecolors=['b', 'g', 'y', 'r'])
            poly.set_alpha(0.5)
            ax.add_collection3d(poly, zs=zs, zdir='y')
            ax.set_title('Lifetime')
            ax.set_xlabel(r'Time [ns]')
            ax.set_yticks(zs)
            ax.set_xticklabels(xs[range(1, 71, 10)], rotation=45, va='baseline', ha='left')
            ax.set_yticklabels(WAVELENGTH_LIFETIME_RANGES, rotation=0, va='baseline', ha='left')
            ax.xaxis.labelpad = 10
            ax.set_xlim3d(0, 64)
            ax.set_ylabel('Channel')
            ax.yaxis.labelpad = 14
            ax.set_ylim3d(0, 3)
            ax.set_zlabel('Intensity')
            ax.set_zlim3d(0, self.lifetime.intensity.max())
            ax.tick_params(axis='x', which='major', pad=13)
            fig.tight_layout()

        else:
            raise KeyError(f'Plot dimensionality {dim} not understood. Should be one of (2D, 3D)')

        ax.set_title('Fluorescence Lifetime')
        return fig, ax

    def plot_scattering_image(self):
        """
        Plots a heatmap with time in the x axis, angle in the y axis and intensity as the color.
        """

        fig = (p9.ggplot(self.scattering)
               + p9.aes('time', 'factor(round(angle, 1))', fill='intensity')
               + p9.geom_tile()
               + p9.scale_fill_continuous('inferno')
               + p9.theme_classic()
               + p9.labs(x='Time (μs)', y='Angle (º)')).draw()
        ax = fig.get_axes()[0]
        ax.set_title('Scattering Image')
        return fig, ax

    def plot_spectrum(self, dim: str = '2D', aggregate: bool = False, k: Optional[int] = None, dpi: int = 120,
                      figsize: Tuple[int, int] = None, view_angle: Tuple[int, int] = None, **kwargs):
        """
        Plots the emission spectrum of the particle either aggregated or split by each time acquisition

        Parameters
        ----------
        dim
            Either 2D or 3D
        aggregate
            Whether to plot the aggregated data (True) or split it by time of acquisition (False). Defaults to False
        k
            If aggregation is made, K is the parameter of the Univariate Spline used to interpolate the data
        dpi
           Dots per inch of the figure
        figsize
            Tuple specifying width and height of the figure in inches
        view_angle
            Tuple specifying the vertical and horizontal rotation angles of the 3D view
        kwargs
            Keyword arguments passed onto `plt.figure` call
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot
        """

        if dim.lower() == '2d':
            figsize = figsize if figsize is not None else (8, 5)
            fig = plt.figure(dpi=dpi, figsize=figsize, **kwargs)
            ax = plt.gca()

            if aggregate:
                k = k if k is not None else 1
                aggregated_intensities = self.spectrum_time_matrix.sum(axis=1)
                x_interpolate = np.linspace(SPECTRAL_WAVELENGTHS.min(), SPECTRAL_WAVELENGTHS.max(), 1000)
                interpolated_intensities = InterpolatedUnivariateSpline(SPECTRAL_WAVELENGTHS,
                                                                        aggregated_intensities, k=k)(x_interpolate)
                ax.plot(x_interpolate, interpolated_intensities, ls='--', color='black',
                        label=f'Interpolated Values (k={k})', alpha=.8)
                ax.plot(SPECTRAL_WAVELENGTHS, aggregated_intensities, 'go', label='Measured Values', alpha=.8)
                ax.legend()
                ax.set_ylabel(r'$\sum_{t=0}^{t=3.5}$ Intensity')
                ax.set_title('Aggregated Spectrum')

            else:
                self.spectrum_time_matrix.plot(ax=ax)
                ax.legend(title=r'Time [$\mu s$]', labels=SPECTRAL_TIME)
                ax.set_title('Particle Emission Spectrum')
                ax.set_ylabel('Intensity')

            ax.set_xlabel(r'Wavelength $(\lambda) [nm]$')

        elif dim.lower() == '3d':
            figsize = figsize if figsize is not None else (8, 6)
            fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
            ax = fig.gca(projection='3d')
            sample = []
            # We will pad the ys and xs so that the polygon is filled starting at the plane y=0
            xs = np.concatenate([[349], SPECTRAL_WAVELENGTHS.round(), [801]])
            zs = SPECTRAL_TIME
            for _, df in self.spectral_data.groupby('time'):
                ys = np.pad(df.intensity.astype(float).values, 1)
                sample.append((list(zip(xs, ys))))
            poly = PolyCollection(sample)
            poly.set_alpha(0.5)
            ax.add_collection3d(poly, zs=zs, zdir='y')
            ax.set_title('Spectrum')
            ax.set_xlabel(r'Wavelength [$\lambda$] (nm)')
            ax.set_xticks(xs)
            ax.set_xticklabels(list(map(int, xs.round())), rotation=45, va='baseline', ha='left')
            ax.xaxis.labelpad = 10
            ax.set_xlim3d(SPECTRAL_WAVELENGTHS.min(), SPECTRAL_WAVELENGTHS.max())
            ax.set_ylabel(r'Time [$\mu$s]')
            ax.set_ylim3d(SPECTRAL_TIME.min(), SPECTRAL_TIME.max())
            ax.set_zlabel('Intensity')
            ax.set_zlim3d(0, self.max_intensity)
            ax.tick_params(axis='x', which='major', pad=13)
            if view_angle is not None:
                ax.view_init(view_angle[0], view_angle[1])
            fig.tight_layout()

        else:
            raise KeyError(f'Plot dimensionality {dim} not understood. Should be one of (2D, 3D)')

        return fig, ax

    def __str__(self):
        repr_str = f'Particle measured at date {self.timestamp}.\nEstimated size of {round(self.size, 2)} μm, ' \
                   f'a max fluorescence intensity of {self.max_intensity} at a wavelength of {self.max_wavelength} nm' \
                   f' measured {self.max_time} μs after excitation.'

        return repr_str

    def __repr__(self):
        return str(self)


class AerosolParticlesData(object):
    def __init__(self, data):
        self.particles = [ParticleData(d) for d in tqdm(data, desc='Processing Particles', leave=False)]
        self.lifetime = pd.DataFrame([d['Lifetime'] for d in data])
        self.spectra = pd.DataFrame([d['Spectrometer'] for d in data])
        self.size = pd.Series([d['Size'] for d in data], name='size')
        self.timestamp = pd.to_datetime(pd.Series([d['Timestamp'] for d in data], name='timestamp'))
        self.max_intensities = (pd.concat([p.max_intensity_properties for p in tqdm(self.particles,
                                                                                    desc='Fetching Max Intensities',
                                                                                    leave=False)])
                                .reset_index(drop=True)
                                .assign(fluorescent=lambda dd: dd['intensity'] >= FLUORESCENCE_THRESHOLD))

    @property
    def n_particles(self):
        return len(self.particles)

    @property
    def aggregated_spectra(self):
        return pd.concat([p.spectrum_time_matrix.sum(axis=1) for p in self.particles], axis=1).sum(axis=1)

    @property
    def summary_df(self):
        return pd.concat([self.timestamp, self.size, self.max_intensities], axis=1)

    @classmethod
    def from_json(cls, filepath):
        with open(filepath, 'r') as fh:
            data = json.load(fh)['Data']
        return cls(data)

    @classmethod
    def from_folder(cls, path):
        return cls.merge([cls.from_json(fn) for fn in tqdm(glob(os.path.join(path, '*.json')),
                                                     desc='Processing files',
                                                     leave=False)])
    def plot_history(self, frequency: str = 'min'):
        """
        Plots the total particles measured by time and split by fluorescent/non-fluorescent.

        Parameters
        ----------
        frequency
            Method for resampling passed to `pandas.DataFrame.resample`. Defaults to 1 min.
        Returns
        -------
        matplotlib.figure.Figure
            Figure and axes containing the plot.

        """
        ax = (self.summary_df
              .set_index('timestamp')
              .resample(frequency)
              ['fluorescent']
              .agg(['sum', 'size'])
              .assign(non_fluorescent=lambda dd: dd['size'] - dd['sum'])
              .rename(columns={'sum': 'Fluorescent', 'non_fluorescent': 'Non-Fluorescent'})
              .drop(columns='size')
              .plot(kind='bar', stacked=True))

        fig = ax.get_figure()
        ax.set_xlabel(None)
        ax.set_ylabel('Number of particles')
        fig.tight_layout()
        return fig, ax

    def plot_spectra(self, k=1, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        agg_spectra = self.aggregated_spectra
        # Discuss need for interpolation. Splines with k > 1 seem to overfit and k=1 is equal to not interpolating
        x_interpolate = np.linspace(SPECTRAL_WAVELENGTHS.min(), SPECTRAL_WAVELENGTHS.max(), 100)
        interpolated_intensities = InterpolatedUnivariateSpline(SPECTRAL_WAVELENGTHS, agg_spectra, k=k)(x_interpolate)
        ax.plot(SPECTRAL_WAVELENGTHS, interpolated_intensities, ls='--', color='black',
                label=f'Interpolated Values (k={k})', alpha=.8)
        ax.plot(SPECTRAL_WAVELENGTHS, agg_spectra, 'go', label='Measured Values', alpha=.8)
        ax.legend()
        ax.set_ylabel(r'$\sum_{p=0}^{p=P}\sum_{t=0}^{t=3.5}$ Intensity$_{pt\lambda}$')
        ax.set_xlabel(r'Wavelength ($\lambda$) [nm]')
        ax.set_title('Aggregated Spectra')
        return fig, ax

    def plot_size(self):
        fig = (p9.ggplot(self.size.to_frame())
               + p9.aes('size')
               + p9.geom_histogram(color='black', bins=20)
               + p9.labs(x='Size (μm)', y='Number of Particles', title='Size distribution')
               + p9.theme_bw()
               ).draw()
        ax = fig.get_axes()[0]

        return fig, ax

    def plot_particle_space(self, domain='time', fig=None, ax2d=None):
        if fig is None and ax2d is None:
            fig, ax2d = plt.subplots()
        scaled_size = self.summary_df['size'].apply(lambda x: np.sqrt(x) * 20)
        if domain.lower() == 'size':
            ax = ax2d.scatter(x='size', y='intensity', c='wavelength', alpha=.3, cmap='viridis', data=self.summary_df,
                              label='', picker=True)
            fig.colorbar(ax, label='Wavelength of max intensity [nm]')
            ax2d.axvline(x=1, linestyle='--', color='lightgrey', label='PM1')
            ax2d.axvline(x=2.5, linestyle='--', color='grey', label='PM2.5')
            ax2d.axvline(x=10, linestyle='--', color='k', label='PM10')
            ax2d.set_xlabel('Size [μm]')
            ax2d.set_ylabel('Intensity')
            legend = plt.legend()
            legend.set_draggable(True)

        elif domain.lower() == 'wavelength':
            ax = ax2d.scatter(x='wavelength', y='intensity', c='size', s=scaled_size,
                              alpha=.3, cmap='viridis', data=self.summary_df, picker=True)
            fig.colorbar(ax, label='Size [μm]')
            ax2d.set_xlabel('Wavelength of max intensity [nm]')
            ax2d.set_ylabel('Intensity')

        elif domain.lower() == 'time':
            ax = ax2d.scatter(x='index', y='intensity', c='wavelength', s=scaled_size,
                              alpha=.3, cmap='viridis', data=self.summary_df.reset_index(), picker=True)
            fig.colorbar(ax, label='Wavelength of max intensity [nm]')
            ax2d.set_xlabel('Particle Index (sorted by measured time)')
            ax2d.set_ylabel('Intensity')

        else:
            raise KeyError(f'Domain {domain} not understood. Must be one of "time", "size", "wavelength".')

        fig.tight_layout()

        return fig, ax

    def plot_particle_space_3d(self, **kwargs):
        fig = plt.figure(**kwargs)
        ax = fig.gca(projection='3d')
        p = ax.scatter(self.summary_df['size'], self.summary_df['wavelength'], self.summary_df['intensity'],
                       c=self.summary_df['time'], cmap='viridis', picker=True)
        ax.set_xlabel('Size [μm]')
        ax.set_ylabel(r'Wavelength $\lambda$ [nm]')
        ax.set_zlabel('Intensity')
        ax.set_title('Particle Space')
        fig.colorbar(p, label='Time of max Intensity (μs)')
        fig.tight_layout()

        return fig, ax

    def filter(self, query: str):
        """
        Filters a collection of particles based on size, intensity or wavelength of max intensity

        Parameters
        ----------
        query
            String with the query that will be passed onto `pd.DataFrame.query` on the instance `summary_df`.
        Returns
        -------
        AerosolParticlesData
            Filtered copy of the original instance.

        """
        print(f'Filtering particles with query: {query}')
        filtered = deepcopy(self)
        # this index is not the same as the list index of the particles, but it
        # is mantained in the summary_df and all other dataframes
        filtered_idxs = filtered.summary_df.query(query).index
        # now to get the list indexes of the particles we are going to do something
        # a bit convoluted, but it works
        filtered_eidxs = (filtered.summary_df
                          .eval(f'q={query}')
                          .assign(eidx=lambda dd: range(len(dd)))
                          .query('q')
                          .eidx)
        filtered.particles = [p for i, p in enumerate(filtered.particles) if i in filtered_eidxs]
        filtered.lifetime = filtered.lifetime.loc[filtered_idxs]
        filtered.spectra = filtered.spectra.loc[filtered_idxs]
        filtered.size = filtered.size.loc[filtered_idxs]
        filtered.timestamp = filtered.timestamp.loc[filtered_idxs]
        filtered.max_intensities = filtered.max_intensities.loc[filtered_idxs]

        return filtered
    
    @classmethod
    def merge(cls, particles_data_list):
        merged = deepcopy(particles_data_list[0])
        for particle_data in particles_data_list[1:]:
            merged.particles.extend(particle_data.particles)
            merged.size = pd.concat([merged.size, particle_data.size]).reset_index(drop=True)
            merged.timestamp = pd.concat([merged.timestamp, particle_data.timestamp]).reset_index(drop=True)
            merged.spectra = pd.concat([merged.spectra, particle_data.spectra]).reset_index(drop=True)
            merged.lifetime = pd.concat([merged.lifetime, particle_data.lifetime]).reset_index(drop=True)
            merged.max_intensities = pd.concat([merged.max_intensities, particle_data.max_intensities]).reset_index(drop=True)
        return merged

    def __iter__(self):
        return iter(self.particles)

    def __getitem__(self, item):
        return self.particles[item]

    def __str__(self):
        return f'Collection of {self.n_particles} aerosol particles measured from ' \
               f'{self.timestamp.min():%Y-%m-%d %H:%M:%S} to {self.timestamp.max():%Y-%m-%d %H:%M:%S}.'

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.particles)

    def __add__(self, other):
        merged = deepcopy(self)
        merged.particles = self.particles + other.particles
        merged.size = pd.concat([self.size, other.size]).reset_index(drop=True)
        merged.timestamp = pd.concat([self.timestamp, other.timestamp]).reset_index(drop=True)
        merged.spectra = pd.concat([self.spectra, other.spectra]).reset_index(drop=True)
        merged.lifetime = pd.concat([self.lifetime, other.lifetime]).reset_index(drop=True)
        merged.max_intensities = pd.concat([self.max_intensities, other.max_intensities]).reset_index(drop=True)
        return merged
