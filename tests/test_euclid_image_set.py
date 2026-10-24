"""Diagnostic images reconstruct fitted Euclid targets without another fit."""
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from tractor import NanoMaggies, PointSource, RaDecPos, Tractor

from euclid_phot import pipeline
from euclid_phot.euclid.cutouts import Cutout
from euclid_phot.euclid.image import build_tractor_image
from euclid_phot.sources import clone_sources
from euclid_phot.units import UJY_PER_NMGY
from euclid_phot.viz import ujy_per_unit


def scene(prior="VIS", spatial=False):
    wcs=WCS(naxis=2)
    wcs.wcs.ctype=["RA---TAN","DEC--TAN"]
    wcs.wcs.crval=[269.48,67.3];wcs.wcs.crpix=[13,13]
    wcs.wcs.cdelt=[-0.1/3600,0.1/3600]
    ra,dec=wcs.pixel_to_world_values([10.,16.],[12.,12.])
    yy,xx=np.mgrid[-3:4,-3:4]
    stamp=np.exp(-(xx**2+yy**2)/2);stamp/=stamp.sum()
    psf=dict(stamps=stamp[None,:,:],ra=np.array([ra[0]]),dec=np.array([dec[0]]),tile_id="123")
    target="H" if prior=="VIS" else "VIS"
    flux={prior:np.array([2.,3.]),target:np.array([7.,11.])}
    sources=[PointSource(RaDecPos(r,d),NanoMaggies(**{prior:f/UJY_PER_NMGY}))
             for r,d,f in zip(ra,dec,flux[prior])]
    mask=np.zeros((25,25),bool);mask[0,0]=True
    cutouts={}
    for band in (prior,target):
        header=fits.Header(dict(MAGZERO=24.5 if band=="VIS" else 30.0,MERTILE="123"))
        c=Cutout(band,np.zeros((25,25)),np.ones((25,25))*.1,wcs,header)
        tim=build_tractor_image(c,stamp,psf_data=psf if spatial else None,pixel_mask=mask)
        truth=clone_sources(sources,band,seed_nmgy=flux[band]/UJY_PER_NMGY)
        c.data[:]=Tractor([tim],truth).getModelImage(0)
        cutouts[band]=c
    result=pipeline.ForcedPhotometryResult(
        target=(269.48,67.3,2.5),prior={"band":prior},
        target_bands={"euclid":(target,)},sources=sources,
        mer_cat=Table(dict(object_id=[1,2])),cutouts=cutouts,
        psf_stamps={prior:stamp,target:stamp},psf_data={prior:psf,target:psf},
        psf_mode="nearest-source-position" if spatial else "field-average",
        flux_quality=np.array([True,False]),fluxes_ujy={prior:flux[prior]},
        prior_pixel_mask=mask)
    run=SimpleNamespace(target_bands={"euclid":(target,)},verbose=False,n_workers=1,
                        persource_psf=spatial)
    return result,run,target,flux


@pytest.mark.parametrize("prior",["VIS","H"])
@pytest.mark.parametrize("spatial",[False,True])
def test_target_images_keep_actual_fitted_fluxes_even_when_catalog_masks_sources(prior,spatial,monkeypatch):
    result,run,target,flux=scene(prior,spatial)
    pipeline._fit_nisp(run,result)
    np.testing.assert_allclose(result.nisp_results[target]["flux_ujy"],flux[target],rtol=1e-5)
    assert np.isnan(result.fluxes_ujy[target][1])
    source_params=[s.getAllParams()[:] for s in result.sources]
    def forbidden(*args,**kwargs):raise AssertionError("image_set must not fit or download")
    monkeypatch.setattr(pipeline,"fit_nisp_forced",forbidden)
    monkeypatch.setattr(pipeline,"query_mer_catalog",forbidden)
    monkeypatch.setattr(pipeline,"fetch_cutout",forbidden)
    images=result.image_set(target)
    f=ujy_per_unit(result.cutouts[target])
    np.testing.assert_allclose(images["model"],result.cutouts[target].data*f,rtol=1e-5,atol=1e-10)
    np.testing.assert_allclose(images["residual"],0,atol=1e-6)
    assert images["invvar"][0,0]==0 and images["wcs"] is result.cutouts[target].wcs
    assert [s.getAllParams()[:] for s in result.sources]==source_params
    # Reconstruct the full scene after coordinate-mode neighbours are split off.
    pipeline._drop_neighbour_rows(result,1)
    assert len(result.sources)==1 and len(result.neighbour_sources)==1
    np.testing.assert_allclose(result.image_set(target)["model"],images["model"])
    assert np.isfinite(result.image_set(prior)["model"]).all()


def test_existing_result_can_render_saved_target_fluxes_without_refitting():
    result,run,target,flux=scene()
    result.fluxes_ujy[target]=flux[target].copy()
    del result.__dict__["nisp_results"]
    images=result.image_set(target)
    np.testing.assert_allclose(images["model"],images["data"],rtol=1e-5,atol=1e-10)
    result.fluxes_ujy[target][1]=np.nan
    with pytest.warns(UserWarning,match="omits 1 source"):
        images=result.image_set(target)
    assert np.isfinite(images["model"]).all()
    assert images["residual"].sum()>0


def test_missing_band_reports_clear_error_and_wise_images_are_unchanged():
    result,run,target,flux=scene()
    with pytest.raises(ValueError,match="no images for 'Y'"):
        result.image_set("Y")
    r=dict(data=np.ones((3,3)),model=np.ones((3,3))*.5,
           residual=np.ones((3,3))*.5,invvar=np.ones((3,3))*4,wcs=result.cutouts["VIS"].wcs)
    result.wise_results={"W1":r}
    images=result.image_set("W1");f=ujy_per_unit(band="W1")
    np.testing.assert_allclose(images["model"],r["model"]*f)
    np.testing.assert_allclose(images["invvar"],r["invvar"]/f**2)


def test_notebook02_diagnostic_cell_accepts_both_completed_prior_runs():
    import json
    from pathlib import Path
    by_prior={}
    for prior in ("VIS","H"):
        result,run,target,flux=scene(prior)
        pipeline._fit_nisp(run,result)
        result.chosen_models=["PointSource"]*2
        w1=np.array([10.,20.])*(1.2 if prior=="H" else 1.)
        result.fluxes_ujy["W1"]=w1
        result.flux_errs_ujy["W1"]=np.ones(2)*.1
        data=np.ones((25,25))
        model=np.ones((25,25))*(.85 if prior=="H" else .8)
        result.wise_results={"W1":dict(data=data,model=model,residual=data-model,
                                      invvar=np.ones((25,25))*4,
                                      wcs=result.cutouts[prior].wcs,chi_inflation=1.)}
        by_prior[prior]=result
    with np.errstate(invalid="ignore",divide="ignore"):
        colour=-2.5*np.log10(by_prior["VIS"].fluxes_ujy["VIS"] / by_prior["VIS"].fluxes_ujy["H"])
    nb=json.loads((Path(__file__).resolve().parents[1]/"notebooks/02_wise_forced_photometry.ipynb").read_text())
    source=next(''.join(c["source"]) for c in nb["cells"]
                if c["cell_type"]=="code" and ''.join(c["source"]).startswith('BAND = "W1"\nimgs ='))
    scope=dict(np=np,by_prior=by_prior,colour=colour)
    exec(compile(source,"02_section8_diagnostics","exec"),scope)
    assert set(scope["imgs"])=={"VIS","H"}
    assert all(set(images)=={"VIS","H","W1"} for images in scope["imgs"].values())
    assert len(scope["rows"])==1 and len(scope["fits"])==1
