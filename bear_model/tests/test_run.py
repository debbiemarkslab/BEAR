import os
from pkg_resources import resource_filename


def test_run_net():
    # Test that the workflow of bear_net works normally
    f_name = resource_filename('bear_model', 'models/config_files/bear_test.cfg')
    script_name = '/'.join(f_name.split('/')[:-2])+'/train_bear_net.py'
    assert 0 == os.system(' '.join(['python3', script_name, f_name]))


def test_run_ref():
    # Test that the workflow of bear_net works normally
    f_name = resource_filename('bear_model', 'models/config_files/bear_test.cfg')
    script_name = '/'.join(f_name.split('/')[:-2])+'/train_bear_ref.py'
    assert 0 == os.system(' '.join(['python3', script_name, f_name]))
