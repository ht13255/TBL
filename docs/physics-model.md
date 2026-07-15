# OpenPhotonTwin physics model and validity regime

OpenPhotonTwin uses SI units throughout: seconds, metres, radians per second,
counts per second, and dimensionless probabilities. The package contains two
different simulation layers. They should not be confused.

- `FockSimulator` evaluates coherent passive linear optics for small photon
  numbers. It is the layer for interference and output occupation probabilities.
- `DigitalTwin` is a chronological Monte Carlo model for source emission,
  hardware timing, routing, loss, control electronics, and detector time tags.
  Event paths do not coherently recombine in this layer.
- `CoherentTimeBinLoop` expands time bins into optical modes and constructs a
  subunitary transfer matrix, retaining interference between injections and
  round trips that arrive in the same bin.

## Gaussian internal mode

The temporal field is

\[
\psi(t)=\frac{1}{(2\pi\sigma_t^2)^{1/4}}
\exp\left[-\frac{(1-iC)(t-t_0)^2}{4\sigma_t^2}\right]
\exp[-i\omega_0(t-t_0)].
\]

Here `temporal_width` is the intensity standard deviation \(\sigma_t\), and
`chirp` is the dimensionless quadratic chirp \(C\). The angular-frequency
intensity width is

\[
\sigma_\omega=\frac{\sqrt{1+C^2}}{2\sigma_t}.
\]

The overlap of two unequal, delayed, detuned, chirped Gaussian modes is
evaluated by the analytic complex Gaussian integral. Jones-vector overlap is
included. `purity` is a phenomenological internal-state purity; the maximum
pair indistinguishability is modeled as \(\sqrt{p_1p_2}\). This captures a
measured HOM purity but is not a complete density matrix for arbitrary mixed
spectral states. For such states, provide a measured positive-semidefinite
`overlap_matrix` directly to `FockSimulator`.

Second-order dispersion multiplies the spectral field by
\(\exp(i\,\mathrm{GDD}\,\Omega^2/2)\). `Wavepacket.dispersed` analytically
updates width and chirp and conserves spectral width and normalization.

## Exact partial distinguishability

For labeled input photons with spatial inputs \(r_j\), detected output list
\(o_j\), transfer amplitudes \(U_j\), and internal Gram matrix
\(S_{jk}=\langle\phi_j|\phi_k\rangle\), the probability is evaluated as

\[
P(\mathbf{s})=\frac{1}{\prod_m s_m!\,\mathcal N}
\sum_{\sigma\in S_n}
\left(\prod_j S_{j,\sigma(j)}\right)
\operatorname{perm}(M_\sigma),
\]

with

\[
(M_\sigma)_{jk}=U_k(o_j,r_k)U_{\sigma(k)}^*(o_j,r_{\sigma(k)}),
\qquad
\mathcal N=\operatorname{perm}
\left[S_{jk}\,\delta_{r_jr_k}\right].
\]

This retains the full pairwise overlap phases and reduces to exact bosonic and
fully distinguishable limits. It is factorial in photon number, so exact
partial distinguishability defaults to at most seven photons. A mean-overlap
approximation remains available only when explicitly requested.

When a circuit contains `SpectralMatrixComponent`, each labeled photon uses
the transfer matrix interpolated at its own wavelength. Measured values must be
passive. Extrapolation outside the calibrated band is rejected by default.

## Correlated pulsed source

`CorrelatedPhotonSource` uses a truncated zero/one/two-photon distribution:

\[
p_2=\frac{g^{(2)}(0)\mu^2}{2},\qquad
p_1=\mu-2p_2,\qquad p_0=1-p_1-p_2.
\]

The requested \(\mu\) and \(g^{(2)}(0)\) are therefore reproduced by the first
two factorial moments as long as these equations define non-negative
probabilities. Collection loss is binomial. On/off blinking is a two-state
Markov chain per excitation pulse. Spectral diffusion is a stationary
Ornstein-Uhlenbeck process with pulse-to-pulse correlation

\[
\rho=\exp[-T_{\rm rep}/\tau_c].
\]

This source truncation is not appropriate when three-photon emission is
non-negligible or when a thermal/SPDC photon-number distribution is required.

## Fiber loop and control

The round-trip survival probability is

\[
\eta_{\rm rt}=\eta_{\rm lumped}
10^{-[\alpha_{\rm dB/km}L_{\rm km}+L_{\rm insertion,dB}]/10}.
\]

The loop applies \(\mathrm{GDD}=\beta_2L\) on every pass. PMD is represented as
a zero-mean arrival-time perturbation with RMS
\(D_{\rm PMD}\sqrt{L}\). Round-trip phase noise is accumulated, while
`PhaseDrift` implements a continuous Wiener process whose increment variance is
proportional to elapsed time. Events at the same time share the same phase.

EOM routing includes finite extinction ratio, insertion transmission, drive
noise, control latency, feed-forward latency/jitter, and a linear voltage rise.
The voltage-to-routing map is phenomenological; measured transfer curves can
instead be supplied with a callable control.

For coherent time-bin evolution, the external and circulating fields obey

\[
y_n=\sqrt{\eta_c}[\sqrt{1-R_n}\,x_n+i\sqrt{R_n}\,\ell_n],
\]

\[
\ell_{n+d}=e^{i\phi_n}\sqrt{\eta_c\eta_{\rm rt}}
[i\sqrt{R_n}\,x_n+\sqrt{1-R_n}\,\ell_n],
\]

where \(d\) is the round-trip delay in time bins. Evaluating this recurrence on
every basis input forms the external transfer matrix. Energy remaining in the
loop beyond the final simulated bin is reported separately from physical loss.

## SNSPD and time tagger

Photon, dark, and afterpulse candidates are processed in true chronological
order. For elapsed time \(\Delta t\) after the hard dead time, efficiency
recovery is

\[
R(\Delta t)=1-\exp[-(\Delta t-t_{\rm dead})/\tau_{\rm rec}].
\]

The photon avalanche probability is the wavelength-dependent system efficiency
times \(R\). Independent pixels have independent recovery clocks. Accepted
avalanches then receive fixed latency, Gaussian readout jitter, an optional
positive exponential jitter tail, and time-tagger quantization. Dark counts
are stationary Poisson candidates. Afterpulsing is a phenomenological branching
process and should be fitted to a specific detector if it matters.

## Calibration and uncertainty

HOM counts are fitted with Poisson maximum likelihood by default. Accidental
background must be measured or estimated independently because background and
intrinsic visibility are not jointly identifiable from a single HOM scan.
Fisher covariance and parametric Poisson bootstrap intervals are available.
Loss estimates include exact Clopper-Pearson intervals before detector-efficiency
correction. Loss position requires ordered physical taps or round-trip-resolved
counts; one input/output pair cannot identify location.

## Important limits

- Passive, linear optics only; no Kerr/Raman/Brillouin nonlinear propagation.
- No master-equation bath or full continuous-frequency entangled joint spectral
  amplitude. Supply measured overlap and S-parameter data when available.
- Event-domain beam splitters sample paths. Use `CoherentTimeBinLoop` or a
  measured expanded-mode transfer matrix for recombining paths.
- Gaussian PMD, Wiener phase noise, OU spectral diffusion, and detector
  afterpulsing are calibrated stochastic models, not microscopic device models.
- Exact Fock calculations scale exponentially; exact general partial
  distinguishability additionally scales factorially.

All stochastic APIs accept a seed. Validation rejects non-passive matrices,
non-positive-semidefinite overlap matrices, invalid probability distributions,
and uncalibrated spectral extrapolation rather than silently normalizing them.
