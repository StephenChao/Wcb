import logging
from dataclasses import dataclass, field
from string import Template

import hist
import numpy as np
from hist import Hist

years = ["2016APV", "2016", "2017", "2018"]

jecs = {
    # "JES": "JES_jes",
    "JER": "JER",
    "Absolute": "split",
    "Absolute_year": "split",
    "BBEC1": "split",
    "BBEC1_year": "split",
    "EC2": "split",
    "EC2_year": "split",
    "FlavorQCD": "split",
    "HF": "split",
    "HF_year": "split",
    "RelativeBal": "split",
    "RelativeSample_year": "split",
    
    #although pdf are added here
    # "pdfscale": "pdfscale",
    
    #although JMS/JMR are added here
    "JMS"     : "JMS",
    "JMR"     : "JMR",
}

@dataclass
class ShapeVar:
    """For storing and calculating info about variables used in fit"""

    name = None
    bins = None  # bin edges
    order_1 = None  # TF order for tf_1, to be decided
    order_2 = None  # TF order for tf_2, to be decided
    order_3 = None  # TF order for tf_3, to be decided


    def __post_init__(self):
        # use bin centers for polynomial fit
        self.pts = self.bins[:-1] + 0.5 * np.diff(self.bins)
        # scale to be between [0, 1]
        self.scaled = (self.pts - self.bins[0]) / (self.bins[-1] - self.bins[0])

@dataclass
class Syst:
    """For storing info about systematics"""

    name = None
    prior = None  # e.g. "lnN", "shape", etc.

    # float if same value in all regions/samples, dictionary of values per region/sample if not
    # if both, region should be the higher level of the dictionary
    value = None
    value_down = None  # if None assumes symmetric effect
    # if the value is different for different regions or samples
    diff_regions = False
    diff_samples = False

    samples = None  # samples affected by it
    samples_corr = True  # if it's correlated between samples, i.e., many samples have the same unc.
    # in case of uncorrelated unc., which years to split into
    uncorr_years = field(default_factory=lambda: years)
    pass_only = False  # is it applied only in the pass regions
    apply_reg = None # if pass_only == True, then apply_reg can be assigned
    
    def __post_init__(self):
        if isinstance(self.value, dict) and not (self.diff_regions or self.diff_samples):
            raise RuntimeError(
                "Value for systematic is a dictionary but neither ``diff_regions`` nor ``diff_samples`` is set."
            )

def add_bool_arg(parser, name, help, default=False, no_name=None):
    """Add a boolean command line argument for argparse"""
    varname = "_".join(name.split("-"))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=varname, action="store_true", help=help)
    if no_name is None:
        no_name = "no-" + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument("--" + no_name, dest=varname, action="store_false", help=no_help)
    parser.set_defaults(**{varname: default})
    
def sum_templates(template_dict, years):
    """Sum templates across years"""

    ttemplate = next(iter(template_dict.values()))  # sample templates to extract values from
    combined = {}

    for region in ttemplate:
        thists = []
        for year in years:
            thists.append(template_dict[year][region])

        combined[region] = sum(thists)
    return combined

def rem_neg(template_dict):
    for _sample, template in template_dict.items():
        template.values()[template.values() < 0] = 0
    #remove negative value
    return template_dict

def get_year_updown(
    templates_dict, sample, region, region_noblinded, blind_str, year, skey, years
):
    """
    Return templates with only the given year's shapes shifted up and down by the ``skey`` systematic.
    Returns as [up templates, down templates]
    """
    updown = []

    for shift in ["up", "down"]:
        sshift = ("%s_%s" % ((skey), (shift)))
        # get nominal templates for each year
        templates = {y: templates_dict[y][region][sample,:] for y in years}
        
        # logging.info(("summed nominal values in %s is %s" % ((year), (sum(list(templates.values())).values()))))

        # replace template only for this year with the shifted template
        if skey in jecs:
            # JEC/JMCs saved as different "region" in dict
            reg_name = ("%s_%s%s" % ((region_noblinded), (sshift), (blind_str)))
            templates[year] = templates_dict[year][reg_name][sample, :]
            # logging.info(("%s values in %s is %s" % ((sshift), (year), (templates[year]))))
        else:
            # weight uncertainties saved as different "sample" in dict
            templates[year] = templates_dict[year][region][("%s_%s" % ((sample), (sshift))), :]
            # logging.info(("%s values in %s is %s" % ((sshift), (year), (templates[year]))))


        # sum templates with year's template replaced with shifted
        updown.append(sum(list(templates.values())).values())

    return updown


def get_effect_updown(values_nominal, values_up, values_down, mask, logger, epsilon):
    effect_up = np.ones_like(values_nominal)
    effect_down = np.ones_like(values_nominal)

    mask_up = mask & (values_up >= 0)
    mask_down = mask & (values_down >= 0)

    effect_up[mask_up] = values_up[mask_up] / values_nominal[mask_up]
    effect_down[mask_down] = values_down[mask_down] / values_nominal[mask_down]

    zero_up = values_up == 0
    zero_down = values_down == 0

    effect_up[mask_up & zero_up] = values_nominal[mask_up & zero_up] * epsilon
    effect_down[mask_down & zero_down] = values_nominal[mask_down & zero_down] * epsilon

    _shape_checks(values_up, values_down, values_nominal, effect_up, effect_down, logger)

    logging.debug(("nominal   : %s" % ((values_nominal))))
    logging.debug(("effect_up  : %s" % ((effect_up))))
    logging.debug(("effect_down: %s" % ((effect_down))))

    return effect_up, effect_down

def _shape_checks(values_up, values_down, values_nominal, effect_up, effect_down, logger):
    norm_up = np.sum(values_up)
    norm_down = np.sum(values_down)
    norm_nominal = np.sum(values_nominal)
    prob_up = values_up / norm_up
    prob_down = values_down / norm_down
    prob_nominal = values_nominal / norm_nominal
    shapeEffect_up = np.sum(
        np.abs(prob_up - prob_nominal) / (np.abs(prob_up) + np.abs(prob_nominal))
    )
    shapeEffect_down = np.sum(
        np.abs(prob_down - prob_nominal) / (np.abs(prob_down) + np.abs(prob_nominal))
    )

    if np.allclose(effect_up, 1.0) and np.allclose(effect_down, 1.0):
        logger.warning("No shape effect")
    elif np.allclose(effect_up, effect_down):
        logger.warning("Up is the same as Down, but different from nominal")
    elif np.allclose(effect_up, 1.0) or np.allclose(effect_down, 1.0):
        logger.warning("Up or Down is the same as nominal (one-sided)")
    elif shapeEffect_up < 0.001 and shapeEffect_down < 0.001:
        logger.warning("No genuine shape effect (just norm)")
    elif (norm_up > norm_nominal and norm_down > norm_nominal) or (
        norm_up < norm_nominal and norm_down < norm_nominal
    ):
        logger.warning("Up and Down vary norm in the same direction")
