#!/usr/bin/perl

# This perl script computes the vertex-facet incidences of the convex polytope given by its
# vertices (row vectors of a matrix in .txt format) and saves them in a *_facets.csv file.

# Load the necessary polymake modules
use application "polytope";

# set the prefered backend using 'reverse search'
prefer_now "lrs";

# get the variable
my $file = shift @ARGV;

# input and output filenames
my $in_path = "< ".$file.".txt";
my $out_path = "> ".$file."_facets.csv";

# load the weightmatrix
open(INPUT, $in_path);
my $matrix = new Matrix<Rational>(<INPUT>);
close(INPUT);

# create the polytope from the matrix
my $polytope = new Polytope<Rational>(VERTICES=>$matrix, LINEALITY_SPACE=>[]);

# compute the vertex-facets incidences of the polytope and export them as .csv file
open OUT, $out_path; print OUT $polytope -> VERTICES_IN_FACETS; close OUT;
