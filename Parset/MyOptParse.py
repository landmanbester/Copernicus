import argparse
import ConfigParser

def readargs():
    conf_parser = argparse.ArgumentParser(
        # Turn off help, so we print all options in response to -h
            add_help=False
            )
    conf_parser.add_argument("-c", "--conf_file",
                             help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    defaults = {
        "nwalkers" : 10,
        "nsamples" : 10000,
        "nburnin"  : 2500,
        "tstar"    : 3.25,
        "doplcf"   : True,
        "dotransform" : True,
        "fname" : "/home/bester/Projects/CP_Dir/",
        "data_prior" : ["H","rho"],
        "data_lik" : ["D","H","dzdw"],
        "zmax" : 2.0,
        "np" : 200,
        "nret" : 100,
        "err" : 1e-5,
        "beta" : 0.01,
        }
    if args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([args.conf_file])
        defaults = dict(config.items('Defaults'))

    # Don't surpress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser],
        # print script description with -h/--help
        description=__doc__,
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    parser.set_defaults(**defaults)
    parser.add_argument("--nwalkers", type=int, help="The number of samplers to spawn")
    parser.add_argument("--nsamples", type=int, help="The number of samples each sampler should draw")
    parser.add_argument("--nburnin", type=int, help="The number of samples in the burnin period")
    parser.add_argument("--tstar", type=float, help="The time up to which to integrate to [in Gpc for now]")
    parser.add_argument("--doplcf", type=bool, help="Whether to compute the interior of the PLC or not")
    parser.add_argument("--dotransform", type=bool, help="Whether to perform the coordinate transformation or not")
    parser.add_argument("--fname", type=str, help="Where to save the results")
    parser.add_argument("--data_prior", type=str, help="The data sets to use to set priors")
    parser.add_argument("--data_lik", type=str, help="The data sets to use for inference")
    parser.add_argument("--zmax", type=float, help="The maximum redshift to go out to")
    parser.add_argument("--np", type=int, help="The number of redshift points to use")
    parser.add_argument("--nret", type=int, help="The number of points at which to return quantities of interest")
    parser.add_argument("--err", type=float, help="Target error of the numerical integration scheme")
    parser.add_argument("--beta", type=float, help="Parameter to control acceptance rate of the MCMC")
    args = parser.parse_args(remaining_argv)


    #return dict containing args
    return vars(args)