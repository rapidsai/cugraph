#!/bin/bash

if [ ! -d "./data" ]
then
    mkdir ./data
fi

cd data

if [ ! -f "./preferentialAttachment.mtx" ]
then
    if [ ! -d "./tmp" ]
    then
        mkdir tmp
        cd tmp

        wget -N https://sparse.tamu.edu/MM/DIMACS10/preferentialAttachment.tar.gz
        wget -N https://sparse.tamu.edu/MM/DIMACS10/caidaRouterLevel.tar.gz
        wget -N https://sparse.tamu.edu/MM/DIMACS10/coAuthorsDBLP.tar.gz
        wget -N https://sparse.tamu.edu/MM/LAW/dblp-2010.tar.gz
        wget -N https://sparse.tamu.edu/MM/DIMACS10/citationCiteseer.tar.gz
        wget -N https://sparse.tamu.edu/MM/DIMACS10/coPapersDBLP.tar.gz
        wget -N https://sparse.tamu.edu/MM/DIMACS10/coPapersCiteseer.tar.gz
        wget -N https://sparse.tamu.edu/MM/SNAP/as-Skitter.tar.gz

        tar xvzf preferentialAttachment.tar.gz
        tar xvzf caidaRouterLevel.tar.gz
        tar xvzf coAuthorsDBLP.tar.gz
        tar xvzf dblp-2010.tar.gz
        tar xvzf citationCiteseer.tar.gz
        tar xvzf coPapersDBLP.tar.gz
        tar xvzf coPapersCiteseer.tar.gz
        tar xvzf as-Skitter.tar.gz

        cd ..

        find ./tmp -name "*.mtx" -exec mv {} . \;

        rm -rf tmp
    fi
fi
