# setup.py
import configparser
import os
import const

config = configparser.RawConfigParser()
setup_path = os.path.dirname(os.path.realpath(__file__))
config.read(setup_path + '/config.ini')
config = config[const.DEFAULT]

