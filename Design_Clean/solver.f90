program solver
real(kind=8) :: thetax, thetay, thetaz
thetax = 0.1d0
thetay = 0.2d0
thetaz = 0.3d0
call Matrix_mult(thetax, thetay, thetaz, R)
print *, 'R = ', R
end program solver
