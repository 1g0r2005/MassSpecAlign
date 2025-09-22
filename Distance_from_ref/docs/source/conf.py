# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Assessment of the quality of MSI data alignment'
copyright = '2025, Basikhin Igor, Andrey Kuzin'
author = 'Basikhin Igor, Andrey Kuzin'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys

def skip_qt_signals(app, what, name, obj, skip, options):
    """Пропустить PyQt сигналы при генерации документации."""
    if 'pyqtSignal' in str(type(obj)):
        return True
    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip_qt_signals)

# Добавляем путь к папке src вашего проекта
sys.path.insert(0, os.path.abspath('../../src'))
extensions = [
    'sphinx.ext.autodoc',    # Для автоматического извлечения docstrings
    'sphinx.ext.viewcode',   # Добавляет ссылки на исходный код
    'sphinx.ext.napoleon',   # Для поддержки Google/NumPy стилей docstrings
    'sphinx.ext.autosummary', # Опционально: для автоматических summary-таблиц
]

templates_path = ['_templates']
exclude_patterns = []

latex_elements = {
    'preamble': r'''
\usepackage{sphinx}
\usepackage{times}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{enumitem}
\usepackage{etoolbox}

% Настройка переноса длинных строк в таблицах
\usepackage{tabularx}
\usepackage{wrapfig}

% Настройка переноса в листингах кода
\usepackage{listings}
\lstset{
    breaklines=true,
    breakatwhitespace=true,
    breakautoindent=true,
    postbreak=\space\space,  % Два пробела после переноса
    columns=fullflexible
}

% Настройка переноса в обычном тексте
\sloppy
\emergencystretch=3em
\hyphenpenalty=10000
\tolerance=9999

'''
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    # 'private-members': False,  # Раскомментируйте, если нужно включать приватные методы
}