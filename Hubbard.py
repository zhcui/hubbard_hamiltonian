#! /usr/bin/env python
"""
script to generate FCIDUMP for Hubbard model.
now support site basis, MO basis and LMO basis (ER localization).
only RHF MO is currently used.
depend on pyscf and libdmet.routine.localizer
(ER localization routine, should be replaced with ER in pyscf in future)

Zhihao Cui
"""

import numpy as np
from pyscf import gto, scf, ao2mo, lo, mcscf
from pyscf import tools
import sys

### parameters:

if len(sys.argv) == 5:
    nelec, U, Nx, Ny = sys.argv[1:]
    nelec = int(nelec)
    U = float(U)
    Nx = int(Nx)
    Ny = int(Ny)
    t = 1.0
elif len(sys.argv) == 3:
    nelec, U = sys.argv[1:] 
    nelec = int(nelec)
    U = float(U)
    Nx, Ny = 2, 4
    t = 1.0
else:
    nelec = 12
    U = 4.0
    Nx, Ny = 4, 4
    t = 1.0

doscf = False
DEBUG = False
np.set_printoptions(3, linewidth =1000)


### function to generate TB hamiltonian of hubbard
def v2idx(n, Ny):
	return [n // Ny, (n%Ny)]

def idx2v(idxvec, Ny):
	return idxvec[0] * Ny + idxvec[1]

def gen_H_tb(t,Nx,Ny,kvec):
    H = np.zeros((Nx,Ny,Nx,Ny),dtype=np.complex)
    for i in range(Nx):
        for j in range(Ny):
            if i == Nx-1:
                H[i,j,0   ,j] += np.exp(-1j*np.dot(np.array(kvec),np.array([Nx,0])))
            else:
                H[i,j,i+1 ,j] += 1

            if i == 0:
                H[i,j,Nx-1,j] += np.exp(-1j*np.dot(np.array(kvec),np.array([-Nx,0])))
            else:
                H[i,j,i-1 ,j] += 1

            if j == Ny-1:
                H[i,j,i,0   ] += np.exp(-1j*np.dot(np.array(kvec),np.array([0,Ny])))
            else:
                H[i,j,i,j+1] += 1

            if j == 0:
                H[i,j,i,Ny-1] += np.exp(-1j*np.dot(np.array(kvec),np.array([0,-Ny])))
            else:
                H[i,j,i,j-1] += 1
    return -t*H.reshape(Nx*Ny, Nx*Ny)


### site basis
norb = Nx * Ny
kvec = [0, 0]
h1 = (gen_H_tb(t, Nx, Ny, kvec)).real
h1[np.abs(h1)> 1e-6] = -1.0
h1[np.abs(h1)<= 1e-6] = 0.0

eri = np.zeros((norb,norb,norb,norb))
for i in range(norb):
    eri[i,i,i,i] = U

tools.fcidump.from_integrals('FCIDUMP.site', h1, eri, norb, \
                                     nelec, ms=0)



### MO basis

mol = gto.M(verbose=4)
mol.nelectron = nelec
mol.incore_anyway = True

mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(norb)
mf._eri = ao2mo.restore(8, eri, norb)

dm0 = np.zeros_like(h1)
np.fill_diagonal(dm0, nelec/float(dm0.shape[0]))


if doscf:
    mf.max_cycle = 50
    mf.conv_tol = 2e-14
else:
    mf.max_cycle = 0
res = mf.run(dm0)
print "scf energy :", res.e_tot
if doscf:
    mf = mf.newton()
    res = mf.run(mf.mo_coeff)
    print "scf energy (newton) :", res.e_tot
print 

rdm = mf.make_rdm1()
print "rdm"
print rdm
print 

mo_coeff = mf.mo_coeff
mo_occ = mf.mo_occ
print "occ"
print mo_occ

mycas = mcscf.CASCI(mf, norb, nelec)
h1e_cas, ecore = mycas.get_h1eff()
h2e_cas = mycas.get_h2eff()

print "ecore", ecore

tools.fcidump.from_integrals('FCIDUMP.MO', h1e_cas, h2e_cas, norb, \
                                     nelec, ms=0)

if DEBUG:
    from pyscf import fci
    e_site, fcivec = fci.direct_spin1.kernel(h1, mf._eri, norb, nelec, verbose=5, \
            conv_tol = 1e-12, max_cycle = 100)

    print "e_site"
    print e_site
    e_MO, fcivec = fci.direct_spin1.kernel(h1e_cas, h2e_cas, norb, nelec, \
            ecore=ecore, verbose=5, conv_tol = 1e-12, max_cycle = 100)
    print "e_MO"
    print e_MO
    print "diff", e_site - e_MO

    assert(abs(e_site - e_MO) < 1e-8)


### LMO basis

def split_localize(orbs, info, h1e, h2e, h0):
    from libdmet.routine.localizer import Localizer

    norbs = h1e.shape[0]
    localorbs = np.zeros_like(orbs) # with respect to original embedding basis
    rotmat = np.zeros_like(h1e) # with respect to active orbitals
    occ, part, virt = info
    h2e =  ao2mo.restore(1, h2e, norbs)
    if occ > 0:
        localizer = Localizer(h2e[:occ, :occ, :occ, :occ])
        print("Localization: occupied")
        localizer.optimize()
        occ_coefs = localizer.coefs.T
        localorbs[:, :occ] = np.dot(orbs[:,:occ], occ_coefs)
        rotmat[:occ, :occ] = occ_coefs
    if virt > 0:
        localizer = Localizer(h2e[-virt:, -virt:, -virt:, -virt:])
        print("Localization: virtual")
        localizer.optimize()
        virt_coefs = localizer.coefs.T
        localorbs[:, -virt:] = np.dot(orbs[:,-virt:], virt_coefs)
        rotmat[-virt:, -virt:] = virt_coefs
    if part > 0:
        localizer = Localizer(h2e[occ:norbs-virt, \
            occ:norbs-virt, occ:norbs-virt, occ:norbs-virt])
        print("Localization: partially occupied:")
        localizer.optimize()
        part_coefs = localizer.coefs.T
        localorbs[:, occ:norbs-virt] = \
                np.dot(orbs[:,occ:norbs-virt], part_coefs)
        rotmat[occ:norbs-virt, occ:norbs-virt] = part_coefs


    h1e_new = rotmat.T.dot(h1e.dot(rotmat))

    h2e_compact = ao2mo.restore(4, h2e, norbs)
    h2e_new = ao2mo.incore.full(h2e_compact, rotmat)
    
    return h1e_new, h2e_new, h0, localorbs, rotmat


nocc = (np.array(mo_occ) > 0).sum()
npart = 0
nvir = (np.array(mo_occ) == 0).sum()
info = (nocc, npart, nvir)
assert(nocc + nvir + npart == norb)

h1e_new, h2e_new, h0, localorbs, rotmat = split_localize(mo_coeff, info, h1e_cas, h2e_cas, ecore)

tools.fcidump.from_integrals('FCIDUMP.LMO', h1e_new, h2e_new, norb, \
                                     nelec, ms=0)
if DEBUG:
    e_LMO, fcivec = fci.direct_spin1.kernel(h1e_new, h2e_new, norb, nelec, \
            ecore=h0, verbose=5, conv_tol = 1e-12, max_cycle = 100)
    print "e_LMO"
    print e_LMO
    print "diff", e_site - e_LMO
    assert(abs(e_site - e_LMO) < 1e-8)


