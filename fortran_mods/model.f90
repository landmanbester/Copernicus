! Subroutines to find Dp
subroutine dDdv(Dp,S)
	implicit none
	real*8, intent(in) :: S
	real*8, intent(out) :: Dp
	Dp = S
end subroutine

subroutine dSdv(Sp,D,rho,u)
	implicit none
	real*8, intent(in) :: D, rho, u
	real*8, intent(out) :: Sp
	real*8, parameter :: PI = 3.141592653589793
	real*8 :: kappa = 8.D0*PI

	Sp = -0.5D0*kappa*D*rho*u**2.D0
end subroutine

subroutine dQdv(Qp,D,S,Q,A,Z,rho,u,Lam,j)
	implicit none
	integer, intent(in) :: j
	real*8, intent(in) :: D, S, Q, A, Z, rho, u, Lam
	real*8, intent(out) :: Qp
	real*8, parameter :: PI = 3.141592653589793
	real*8 :: kappa = 8.D0*PI

	if (j == 1) then
		Qp = 0.0D0
	else
		Qp = (1.0D0 - D*S*Z - 2.0D0*Q*S - A*S**2.0D0 + &
                0.5D0*kappa*rho*D**2.0D0*(A*u**2.0D0 - 1.0D0) - Lam*D**2.0D0)/(2.0D0*D)
	endif
end subroutine

subroutine dAdv(Ap,Z)
	implicit none
	real*8, intent(in) :: Z
	real*8, intent(out) :: Ap

	Ap = Z

end subroutine

subroutine dZdv(Zp,D,S,Q,A,rho,u,Lam,j)
	implicit none
	integer, intent(in) :: j
	real*8, intent(in) :: D, S, Q, A, rho, u, Lam
	real*8, intent(out) :: Zp
	real*8, parameter :: PI = 3.141592653589793
	real*8 :: kappa = 8.D0*PI

	if (j == 1) then
		Zp = kappa*rho/3.D0 - 2.D0*Lam/3.D0
	else
		Zp = kappa*rho + 4.D0*Q*S/D**2.D0 + 2.D0*A*S**2.D0/D**2.D0 - 2.D0/D**2.D0
	endif
end subroutine

subroutine dZdv2(Zp,D,S,Qp,A,Z,rho,u,Lam,j)
	implicit none
	integer, intent(in) :: j
	real*8, intent(in) :: D, S, Qp, A, Z, rho, u, Lam
	real*8, intent(out) :: Zp
	real*8, parameter :: PI = 3.141592653589793
	real*8 :: kappa = 8.D0*PI

	if (j < 2) then
		Zp = kappa*rho/3.D0 - 2.D0*Lam/3.D0
	else
		Zp = kappa*rho*A*u**2.D0 - 4.D0*Qp/D - 2.D0*Z*S/D - 2.D0*Lam
	endif
end subroutine

subroutine drhodw(dotrho,D,S,Q,A,rho,rhop,u,up,j)
	implicit none
	integer, intent(in) :: j
	real*8, intent(in) :: D, S, Q, A, rho, rhop, u, up
	real*8, intent(out) :: dotrho

	real*8, parameter :: PI = 3.141592653589793
	real*8 :: kappa = 8.D0*PI

	if (j <2) then
		dotrho = -3.D0*rho*up
	else
		dotrho = rho*(-up/u**3.D0 - 2.D0*Q/D + S*(1.D0/u**2.D0 - A)/D) + 0.5D0*rhop*(1.D0/u**2.D0 - A)
	endif
end subroutine

subroutine dudw(dotu,A,Z,u,up)
	implicit none
	real*8, intent(in) :: A, Z, u, up
	real*8, intent(out) :: dotu
	dotu = 0.5D0*((1.D0/u**2.D0 - A)*up - Z*u)

end subroutine

subroutine dy1(tp,y2)

	implicit none
	real*8, intent(in) :: y2
	real*8, intent(out) :: tp
	
	tp = y2
	
end subroutine

subroutine dy2(tpp,y2,u,up)

	implicit none
	real*8, intent(in) :: y2,up,u
	real*8, intent(out) :: tpp
	
	tpp = -up*y2**2/u**2
	
end subroutine

subroutine dy3(rp,y4)

	implicit none
	real*8, intent(in) :: y4
	real*8, intent(out) :: rp
	
	rp = y4
	
end subroutine

subroutine dy4(rpp,y2,y4,u,up,X,Xr)

	implicit none
	real*8, intent(in) :: y2,y4,u,up,X,Xr
	real*8, intent(out) :: rpp
	
	rpp = -Xr*y4**2.D0/X - 2.D0*up*y2*y4/u**2.D0
	
end subroutine

subroutine dddw(dotd,d,dp,A,Z,u,up,NI,NJ,i,jmax)
	implicit none
	integer, intent(in) :: NI,NJ,i,jmax
	real*8, dimension(NJ), intent(in) :: d,dp,A,Z,u,up
	real*8, dimension(NJ,NI), intent(inout) :: dotd
	
	dotd(1:jmax,i) = -0.5D0*dp(1:jmax)*(A(1:jmax) - 1.D0/u(1:jmax)**2.D0) &
	- 0.5D0*d(1:jmax)*(Z(1:jmax) + 2.D0*up(1:jmax)/u(1:jmax)**3.D0)

end subroutine

subroutine transt(dtdw,dwdt,dtdv,dvdt,A,u,NI,NJ,i,jmax)
	implicit none
	integer, intent(in) :: NI,NJ,i,jmax
	real*8, dimension(NJ), intent(in) :: A,u
	real*8, dimension(NJ,NI), intent(inout) :: dtdw,dwdt,dtdv,dvdt
	
	dtdw(1:jmax,i) = (A(1:jmax)*u(1:jmax)**2 + 1.D0)/(2.D0*u(1:jmax))
	dwdt(1:jmax,i) = u(1:jmax)
	dtdv(1:jmax,i) = -u(1:jmax)
	dvdt(1:jmax,i) = (A(1:jmax)*u(1:jmax)**2 - 1.D0)/(2.D0*u(1:jmax))
	
end subroutine

subroutine transr(drdw,dwdr,drdv,dvdr,A,u,NI,NJ,i,jmax)
	implicit none
	integer, intent(in) :: NI,NJ,i,jmax
	real*8, dimension(NJ), intent(in) :: A,u
	real*8, dimension(NJ,NI), intent(inout) :: drdw,dwdr,drdv,dvdr

    drdw(1:jmax,i) = drdv(1:jmax,i)*(A(1:jmax)*u(1:jmax)**2.D0 - 1.D0)/(2.D0*u(1:jmax)**2.D0)
    dwdr(1:jmax,i) = -u(1:jmax)**2.D0/drdv(1:jmax,i)
    dvdr(1:jmax,i) = (1.D0+A(1:jmax)*u(1:jmax)**2.D0)/(2.D0*drdv(1:jmax,i))	
	
end subroutine

subroutine getXandXr(X,dXdr,dwdr,dvdr,drdv,drdvp,drdvd,u,up,ud,NI,NJ,i,jmax)
	implicit none
	integer, intent(in) :: NI,NJ,i,jmax
	real*8, dimension(NJ), intent(in) :: dwdr, dvdr, drdv, drdvp,drdvd,u,up,ud
	real*8, dimension(NJ,NI), intent(inout) :: X,dXdr
	
	X(1:jmax,i) = u/drdv
	dXdr(1:jmax,i) = dwdr*(ud/drdv - u*drdvd/drdv**2) + dvdr*(up/drdv - u*drdvp/drdv**2)
	
end subroutine

subroutine get_LLTBCon(LLTBCon,Dww,Aw,D,S,Q,A,Z,Zp,u,rho,Lam,delw,NI,NJ,vmaxi)
	implicit none
	integer, intent(in) :: NI,NJ,vmaxi(NI)
	real*8, intent(in) :: delw, Lam
	real*8, dimension(NJ,NI), intent(in) :: D,S,Q,A,Z,Zp,u,rho
        real*8, dimension(NJ,NI), intent(inout) :: LLTBCon,Dww,Aw
	real*8, parameter :: PI = 3.141592653589793
	real*8 :: kappa = 8.D0*PI	
        integer :: j,jmax,imax,tmp(NI),njf
        !real*8, dimension(NJ,NI) :: Dww, Aw

        Dww = 0.D0
        Aw = 0.D0

        njf = vmaxi(NI)
        j = 1
        imax = NI
        do while (imax > 5 .and. j <= NJ)
            if (j <= njf) then
                call dd5fT(Dww,D,-delw,j,NI,NJ,NI)
                call d5fT(Aw,A,-delw,j,NI,NJ,NI)
            else
                tmp = 0
                where (vmaxi >= j)
                    tmp = 1
                end where
                imax = sum(tmp) !This is the number of PLCs (temporal grid points) within the causal horizon
                call dd5fT(Dww,D,-delw,j,NI,NJ,imax)
                call d5fT(Aw,A,-delw,j,NI,NJ,imax)
            endif
            LLTBCon(j,1:imax) = 0.5D0*A(j,1:imax)*Zp(j,1:imax)*D(j,1:imax) - 2.0D0*Dww(j,1:imax) + Z(j,1:imax)*Q(j,1:imax) &
            - S(j,1:imax)*Aw(j,1:imax) + A(j,1:imax)*Z(j,1:imax)*S(j,1:imax) - 0.25D0*kappa*rho(j,1:imax)*D(j,1:imax)*&
            (1.D0/u(j,1:imax)**2.D0 + u(j,1:imax)**2.D0*A(j,1:imax)**2.D0) + Lam*A(j,1:imax)*D(j,1:imax)
            j = j + 1
        end do
	
end subroutine

subroutine curve_test(T2,D,S,dSdv,u,up,upp,vmaxi,i,NI,NJ)
	implicit none
        !subroutine parameters
	integer, intent(in) :: NI,NJ,vmaxi(NI),i
	real*8, dimension(NJ), intent(in) :: D,S,dSdv,u,up,upp
        real*8, dimension(NJ,NI), intent(inout) :: T2
        !Local parameters
        integer :: njf

        !Compute curve test
        njf = vmaxi(i)
        T2(1:njf,i) = 1.0D0 + up(1:njf)**2.0D0*(u(1:njf)**2.0D0*(D(1:njf)*(dSdv(1:njf)/up(1:njf)**2.0D0 - &
        S(1:njf)*upp(1:njf)/(up(1:njf)**3.0D0)) - (S(1:njf)/up(1:njf))**2.0D0) - D(1:njf)**2.0D0)/u(1:njf)**4.0D0 &
        + (upp(1:njf)/u(1:njf)**3.0D0 - 2.0D0*up(1:njf)**2.0D0/u(1:njf)**4.0D0)*D(1:njf)*(u(1:njf)*S(1:njf)/up(1:njf) + D)

end subroutine


subroutine shear_test(T1,u,up,D,S,Q,A,vmaxi,i,NI,NJ)
	implicit none
        !subroutine parameters
	integer, intent(in) :: NI,NJ,vmaxi(NI),i
	real*8, dimension(NJ), intent(in) :: D,S,Q,A,u,up
        real*8, dimension(NJ,NI), intent(inout) :: T1
        !Local parameters
        integer :: njf

        !Compute shear test
        njf = vmaxi(i)
        T1(1:njf,i) = 1.0D0 - u(1:njf)**3.0D0*(Q(1:njf) - S(1:njf)*(1/u(1:njf)**2.0D0 - A(1:njf))/2.0D0)/(up(1:njf)*D(1:njf))
        T1(1,i) = 0.0D0

end subroutine
