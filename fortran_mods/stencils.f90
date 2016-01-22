! Support for differentiation (finite diff stencils)

subroutine d5f(df,f, h, i, NI, NJ,jmax)
  
implicit none
integer, intent(in) :: NI, NJ, i, jmax
real*8, intent(in) :: f(NJ,NI), h
real*8, intent(inout) :: df(NJ,NI)

integer :: j

do j=1,jmax
  if (j < 3) then
    df(j,i) = (-25.D0*f(j,i) + 48.D0*f(j+1,i) - 36.D0*f(j+2,i) + 16.D0*f(j+3,i) - 3.D0*f(j+4,i))/(12.D0*h)
  else if (j >= 3 .and. j < jmax-2) then
    df(j,i) = (f(j-2,i) - 8.D0*f(j-1,i) + 8.D0*f(j+1,i)-f(j+2,i))/(12.D0*h)
  else if (j >= jmax-2) then
    df(j,i) = -(-25.D0*f(j,i) + 48.D0*f(j-1,i) - 36.D0*f(j-2,i) + 16.D0*f(j-3,i) - 3.D0*f(j-4,i))/(12.D0*h)
  endif
end do

end subroutine  

subroutine dfdv(df,f, h, i, NI, NJ,jmax)
  
implicit none
integer, intent(in) :: NI, NJ, i, jmax
real*8, intent(in) :: f(NJ,NI), h
real*8, intent(inout) :: df(NJ,NI)

integer :: j

do j=1,jmax
  if (j==1) then
    df(j,i) = (-3.D0*f(j,i) + 4.D0*f(j+1,i) - f(j+2,i))/(2.D0*h)
  endif

  if (j==2 .or. j==jmax-1) then
    df(j,i) = (f(j+1,i) - f(j-1,i))/(2.D0*h)
  endif

  if (j >= 3 .and. j <= jmax-2) then
    df(j,i) = (f(j-2,i) - 8.D0*f(j-1,i) + 8.D0*f(j+1,i)-f(j+2,i))/(12.D0*h)
  endif

  if (j==jmax) then
    df(j,i) = -(-3.D0*f(j,i) + 4.D0*f(j-1,i) - f(j-2,i))/(2.D0*h)
  endif

end do
end subroutine

subroutine d2fdv(d2f,f, h, i, NI, NJ,jmax)
  
implicit none
integer, intent(in) :: NI, NJ, i, jmax
real*8, intent(in) :: f(NJ,NI), h
real*8, intent(inout) :: d2f(NJ,NI)

integer :: j

do j=1,jmax
  if (j==1) then
    d2f(j,i) = (2.D0*f(j,i) - 5.D0*f(j+1,i) + 4.D0*f(j+2,i) - f(j+3,i))/(h**2)
  endif

  if (j==2 .or. j==jmax-1) then
    d2f(j,i) = (f(j+1,i) - 2.D0*f(j,i) + f(j-1,i))/(h**2.D0)
  endif

  if (j >= 3 .and. j <= jmax-2) then
    d2f(j,i) = (-f(j-2,i) + 16.D0*f(j-1,i) - 30.d0*f(j,i) + 16.D0*f(j+1,i) - f(j+2,i))/(12.D0*h**2)
  endif

  if (j==jmax) then
    d2f(j,i) = (2.D0*f(j,i) - 5.D0*f(j-1,i) + 4.D0*f(j-2,i) - f(j-3,i))/(h**2)
  endif

end do
end subroutine

subroutine d5f1D(df,f, h, NJ,jmax)
  
implicit none
integer, intent(in) :: NJ, jmax
real*8, intent(in) :: f(NJ), h
real*8, intent(out) :: df(NJ)

integer :: j

do j=1,jmax
  if (j < 3) then
    df(j) = (-25.D0*f(j) + 48.D0*f(j+1) - 36.D0*f(j+2) + 16.D0*f(j+3) - 3.D0*f(j+4))/(12.D0*h)
  elseif (j >= 3 .and. j < jmax-2) then
    df(j) = (f(j-2) - 8.D0*f(j-1) + 8.D0*f(j+1)-f(j+2))/(12.D0*h)
  elseif (j >= jmax-2) then
    df(j) = -(-25.D0*f(j) + 48.D0*f(j-1) - 36.D0*f(j-2) + 16.D0*f(j-3) - 3.D0*f(j-4))/(12.D0*h)
  endif
end do

end subroutine

subroutine dd5f(d2f,f,h,i,NI,NJ,jmax)
	implicit none
	integer, intent(in) :: i, NI, NJ, jmax
	real*8, intent(in) :: f(NJ,NI), h
	real*8, intent(out) :: d2f(NJ,NI)
	
	integer :: j
	do j=1,jmax
		if (j < 3) then
			d2f(j,i) = (45.D0*f(j,i) - 154.D0*f(j+1,i) + 214.D0*f(j+2,i) - 156.D0*f(j+3,i) + 61.D0*f(j+4,i) - 10.D0*f(j+5,i))/(12.D0*h**2)
		elseif (j >= 3 .and. j < jmax-2) then
			d2f(j,i) = (-f(j-2,i) + 16.D0*f(j-1,i) - 30.d0*f(j,i) + 16.D0*f(j+1,i) - f(j+2,i))/(12.D0*h**2)
		elseif (j >= jmax-2) then
			d2f(j,i) = (45.D0*f(j,i) - 154.D0*f(j-1,i) + 214.D0*f(j-2,i) - 156.D0*f(j-3,i) + 61.D0*f(j-4,i) - 10.D0*f(j-5,i))/(12.D0*h**2)
		endif
	end do
end subroutine

subroutine dd5fT(d2f,f,h,j,NI,NJ,imax)
	implicit none
	integer, intent(in) :: j, NI, NJ, imax
	real*8, intent(in) :: f(NJ,NI), h
	real*8, intent(out) :: d2f(NJ,NI)
	
	integer :: i
	do i=1,imax
		if (i < 3) then
			d2f(j,i) = (45.D0*f(j,i) - 154.D0*f(j,i+1) + 214.D0*f(j,i+2) - 156.D0*f(j,i+3) + 61.D0*f(j,i+4) - 10.D0*f(j,i+5))/(12.D0*h**2)
		elseif (i >= 3 .and. i < imax-2) then
			d2f(j,i) = (-f(j,i-2) + 16.D0*f(j,i-1) - 30.d0*f(j,i) + 16.D0*f(j,i+1) - f(j,i+2))/(12.D0*h**2)
		elseif (i >= imax-2) then
			d2f(j,i) = (45.D0*f(j,i) - 154.D0*f(j,i-1) + 214.D0*f(j,i-2) - 156.D0*f(j,i-3) + 61.D0*f(j,i-4) - 10.D0*f(j,i-5))/(12.D0*h**2)
		endif
	end do
end subroutine

subroutine d5fT(df,f, h, j, NI, NJ,imax)
  
implicit none
integer, intent(in) :: NI, NJ, j, imax
real*8, intent(in) :: f(NJ,NI), h
real*8, intent(inout) :: df(NJ,NI)

integer :: i

do i=1,imax
  if (i < 3) then
    df(j,i) = (-25.D0*f(j,i) + 48.D0*f(j,i+1) - 36.D0*f(j,i+2) + 16.D0*f(j,i+3) - 3.D0*f(j,i+4))/(12.D0*h)
  else if (i >= 3 .and. i < imax-2) then
    df(j,i) = (f(j,i-2) - 8.D0*f(j,i-1) + 8.D0*f(j,i+1)-f(j,i+2))/(12.D0*h)
  else if (i >= imax-2) then
    df(j,i) = -(-25.D0*f(j,i) + 48.D0*f(j,i-1) - 36.D0*f(j,i-2) + 16.D0*f(j,i-3) - 3.D0*f(j,i-4))/(12.D0*h)
  endif
end do

end subroutine  

subroutine dd5f1D(d2f,f,h,NJ,jmax)
	implicit none
	integer, intent(in) :: NJ, jmax
	real*8, intent(in) :: f(NJ), h
	real*8, intent(out) :: d2f(NJ)
	
	integer :: j
	do j=1,jmax
		if (j < 3) then
			d2f(j) = (45.D0*f(j) - 154.D0*f(j+1) + 214.D0*f(j+2) - 156.D0*f(j+3) + 61.D0*f(j+4) - 10.D0*f(j+5))/(12.D0*h**2)
		elseif (j >= 3 .and. j < jmax-2) then
			d2f(j) = (-f(j-2) + 16.D0*f(j-1) - 30.d0*f(j) + 16.D0*f(j+1) - f(j+2))/(12.D0*h**2)
		elseif (j >= jmax-2) then
			d2f(j) = (45.D0*f(j) - 154.D0*f(j-1) + 214.D0*f(j-2) - 156.D0*f(j-3) + 61.D0*f(j-4) - 10.D0*f(j-5))/(12.D0*h**2)
		endif
	end do
end subroutine
