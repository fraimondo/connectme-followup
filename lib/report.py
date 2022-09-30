from nice_ext.api.modules import register_module
from nice_ext.api import create_report


def register():
    register_module('report', 'copenhagen/rs', _create_rs_report)


def _create_rs_report(instance, report, config_params):
    n_channels = config_params.get('n_channels', 25)

    report_params = {
        'reduction_rs': f'copenhagen/rs/{n_channels}/trim_mean80',
    }
    config_params.update(report_params)
    # Fit report
    return create_report(instance, config='icm/rs',
                         config_params=config_params, report=report)
