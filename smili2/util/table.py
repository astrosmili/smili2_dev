#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is a submodule of smili. This module saves some common functions,
variables, and data types in the smili module.
'''
from pandas import DataFrame, Series
from numpy import zeros, asarray


class TableHeader(DataFrame):
    name2dtype = None
    name2unit = None

    def __init__(self, data=zeros([0, 4])):
        super().__init__(data=data, columns=[
            "name", "dtype", "unit", "comment"])
        self._create_name2dtype()
        self._create_name2unit()

    def _create_name2dtype(self):
        self.name2dtype = {}
        for i in range(len(self)):
            self.name2dtype[self.loc[i, "name"]] = self.loc[i, "dtype"]

    def _create_name2unit(self):
        from .units import Unit
        self.name2unit = {}
        for i in range(len(self)):
            self.name2unit[self.loc[i, "name"]] = Unit(self.loc[i, "unit"])

    @property
    def _constructor(self):
        return TableHeader

    @property
    def _constructor_sliced(self):
        return TableHeaderSeries


class TableHeaderSeries(Series):
    @property
    def _constructor(self):
        return TableHeaderSeries

    @property
    def _constructor_expanddim(self):
        return TableHeader


class DataTable(DataFrame):
    '''
    This is a class describing common variables and methods of VisTable,
    BSTable and CATable.
    '''
    header = TableHeader()

    def convert_format(self):
        columns = self.header.name.to_list()
        dtypes = self.header.dtype.to_list()

        for i in range(len(columns)):
            self[columns[i]] = asarray(
                self[columns[i]].values, dtype=dtypes[i])

    @property
    def _constructor(self):
        return DataTable

    @property
    def _constructor_sliced(self):
        return DataSeries


class DataSeries(Series):
    @property
    def _constructor(self):
        return DataSeries

    @property
    def _constructor_expanddim(self):
        return DataTable
