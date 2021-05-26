import os
from pkg_resources import resource_filename
from bear_model.models import train_bear_net
from bear_model.models import train_bear_ref
import configparser

def test_run_net():
#     Test that the workflow of bear_net works normally
    f_name = resource_filename('bear_model', 'models/config_files/bear_test.cfg')
#     script_name = resource_filename('bear_model', 'models/train_bear_net.py')
    config = configparser.ConfigParser()
    config.read(f_name)
    assert 1 == train_bear_net.main(config)


def test_run_ref():
#     Test that the workflow of bear_net works normally
    f_name = resource_filename('bear_model', 'models/config_files/bear_test.cfg')
#     script_name = resource_filename('bear_model', 'models/train_bear_ref.py')
    config = configparser.ConfigParser()
    config.read(f_name)
    assert 1 == train_bear_ref.main(config)
