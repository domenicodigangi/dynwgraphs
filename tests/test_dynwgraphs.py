#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic tests for dynwgraphs package.

@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday June 5th 2021
"""

import pytest
from click.testing import CliRunner

from dynwgraphs import dynwgraphs
from dynwgraphs import cli


class TestCLI:
    """Test command line interface."""

    def test_command_line_interface(self):
        """Test the CLI main command."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'dynwgraphs.cli.main' in result.output

    def test_command_line_help(self):
        """Test the CLI help command."""
        runner = CliRunner()
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output


class TestPackageImport:
    """Test that package imports work correctly."""

    def test_import_dynwgraphs(self):
        """Test that the main package can be imported."""
        import dynwgraphs
        assert dynwgraphs is not None

    def test_import_dirSpW1(self):
        """Test that dirSpW1 model can be imported."""
        from dynwgraphs import dirSpW1_dynNet_SD
        assert dirSpW1_dynNet_SD is not None

    def test_import_dirBin1(self):
        """Test that dirBin1 model can be imported."""
        from dynwgraphs.dirBin1_dynNets import dirBin1_dynNet_SD
        assert dirBin1_dynNet_SD is not None

    def test_import_utils(self):
        """Test that utils can be imported."""
        from dynwgraphs.utils import tens
        assert tens is not None
