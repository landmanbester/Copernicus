subroutine predict(delw,D,S,Q,A,Z,rhop,up,NI,NJ,jmax,i,rho,u,rhod,ud)
	implicit none
	!Subroutine parameters
	integer, intent(in) :: NJ,NI,jmax,i
	real*8, intent(in) :: delw
	real*8, dimension(NJ), intent(in) :: D,S,Q,A,Z,rhop,up
        !real*8, dimension(NI), intent(in) :: rhoow
	real*8, dimension(NJ,NI), intent(in) :: rhod,ud
	real*8, dimension(NJ,NI), intent(inout) :: rho,u
	!Local variables
	integer :: j
	real*8 :: dotu, dotrho

	do j=1,jmax
		if (i==2) then
			if (j==1) then
				u(j,i) = 1.D0
                                rho(j,i) = rho(j,i-1) - delw*rhod(j,i-1) !rho(j,i) = rhoow(i)
			else
				!call dudw(dotu,A(j),Z(j),u(j,i-1),up(j))
				u(j,i) = u(j,i-1) - delw*ud(j,i-1)
                                !call drhodw(dotrho,D(j),S(j),Q(j),A(j),rho(j,i-1),rhop(j),u(j,i-1),up(j),j)
                                rho(j,i) = rho(j,i-1) - delw*rhod(j,i-1)		
			endif
		else
			if (j==1) then
				u(j,i) = 1.D0
                                rho(j,i) = rho(j,i-1) - delw*(3.D0*rhod(j,i-1) - rhod(j,i-2))/2.D0 !rho(j,i)rho(j,i) = rhoow(i)
			else
				!call dudw(dotu,A(j),Z(j),u(j,i-1),up(j))
				u(j,i) = u(j,i-1) - delw*(3.D0*ud(j,i-1) - ud(j,i-2))/2.D0
                                !call drhodw(dotrho,D(j),S(j),Q(j),A(j),rho(j,i-1),rhop(j),u(j,i-1),up(j),j)
                                rho(j,i) = rho(j,i-1) - delw*(3.D0*rhod(j,i-1) - rhod(j,i-2))/2.D0			
			endif	
		endif
	end do
end subroutine

!---------------------------------------------------------------------------
! Subroutine to solve hypersurface equations and evaluate derivatives of fluid variables
!---------------------------------------------------------------------------
subroutine evaluate(rho,u,rhod,ud,Lam,delv,D,S,Q,A,Z,rhop,up,upp,NI,NJ,jmax,i,dSdvp,dQdvp,dZdvp)
	implicit none
	!Subroutine parameters
	integer, intent(in) :: NI,NJ,jmax,i
	real*8, intent(in) :: delv, Lam
	!real*8, dimension(NI), intent(in) :: upow
	real*8, dimension(NJ,NI), intent(inout) :: D,S,Q,A,Z,rhop,up,upp,rho,u,rhod,ud,dSdvp,dQdvp,dZdvp
	!Local variables
	integer :: j, k
	real*8 :: pD,pS,pQ,pA,pZ  !Predicted values
	real*8 :: dD1,dS1,dQ1,dA1,dZ1 !Previous derivatives
	real*8 :: dD2,dS2,dQ2,dA2,dZ2 !Predicted derivatives
	real*8 :: dotrho,dotu,errQ,errA,errZ,tol,pQs,pAs,pZs,maxit

        !Set tolerance for PC step (for some reason the code is extremely sensitive to errors in Q,A and Z)
        tol = 1.0E-9
        maxit = 100000 !Maximum number of iterations

	!Init local arrays
	pD = 0.D0
	pS = 0.D0
	pQ = 0.D0
	pA = 0.D0
	pZ = 0.D0
	dD1 = 0.D0
	dS1 = 0.D0
	dQ1 = 0.D0
	dA1 = 0.D0
	dZ1 = 0.D0
	dD2 = 0.D0
	dS2 = 0.D0
	dQ2 = 0.D0
	dA2 = 0.D0
	dZ2 = 0.D0
	
	!Enforce IC's
	D(1,i) = 0.D0
	S(1,i) = 1.D0
	Q(1,i) = 0.D0
	A(1,i) = 1.D0
	Z(1,i) = 0.D0

	!Get radial derivs of fluid varables
	call d5f(rhop,rho,delv,i,NI,NJ,jmax)
	call d5f(up,u,delv,i,NI,NJ,jmax)	
	call dd5f(upp,u,delv,i,NI,NJ,jmax)	
	
	!First we solve for D and S (This will result in more accurate reconstructions of Q,A and Z)
	do j=2,jmax
		if (j == 2) then
			!Evaluate derivs on previous step
			call dDdv(dD1,S(j-1,i))
			call dSdv(dS1,D(j-1,i),rho(j-1,i),u(j-1,i))
                        dSdvp(j-1,i) = dS1	
			!Predict values on next step
			pD = D(j-1,i) + delv*dD1
			pS = S(j-1,i) + delv*dS1
			!Evaluate derivs with predicted values
			call dDdv(dD2,pS)
			call dSdv(dS2,pD,rho(j,i),u(j,i))
			!Correct
			pD = D(j-1,i) + 0.5D0*delv*(dD1 + dD2)
			pS = S(j-1,i) + 0.5D0*delv*(dS1 + dS2)			
			!Evaluate derivs with corrected values
			call dDdv(dD2,pS)
			call dSdv(dS2,pD,rho(j,i),u(j,i))
			!Correct
			D(j,i) = D(j-1,i) + 0.5D0*delv*(dD1 + dD2)
			S(j,i) = S(j-1,i) + 0.5D0*delv*(dS1 + dS2)
		elseif (j>2 .and. j<=4) then
			!Evaluate derivs on previous step
			call dDdv(dD1,S(j-1,i))
			call dSdv(dS1,D(j-1,i),rho(j-1,i),u(j-1,i))
                        dSdvp(j-1,i) = dS1	
			!Predict values on next step ()
			pD = D(j-1,i) + delv*(3.D0*dD1 - S(j-2,i))/2.D0
			pS = S(j-1,i) + delv*(3.D0*dS1 - dSdvp(j-2,i))/2.D0
			!Evaluate derivs with predicted values
			call dDdv(dD2,pS)
			call dSdv(dS2,pD,rho(j,i),u(j,i))
			!Correct
			pD = D(j-1,i) + 0.5D0*delv*(dD1 + dD2)
			pS = S(j-1,i) + 0.5D0*delv*(dS1 + dS2)			
			!Evaluate derivs with predicted values
			call dDdv(dD2,pS)
			call dSdv(dS2,pD,rho(j,i),u(j,i))
			!Correct
			D(j,i) = D(j-1,i) + 0.5D0*delv*(dD1 + dD2)
			S(j,i) = S(j-1,i) + 0.5D0*delv*(dS1 + dS2)
                else
			!Evaluate derivs on previous step
			call dDdv(dD1,S(j-1,i))
			call dSdv(dS1,D(j-1,i),rho(j-1,i),u(j-1,i))
                        dSdvp(j-1,i) = dS1	
			!Predict values on next step
			pD = D(j-1,i) + delv*(55.0D0*dD1 - 59.0D0*S(j-2,i) + 37.0D0*S(j-3,i) - 9.0D0*S(j-4,i))/24.D0
			pS = S(j-1,i) + delv*(55.0D0*dS1 - 59.0D0*dSdvp(j-2,i) + 37.0D0*dSdvp(j-3,i) - 9.0D0*dSdvp(j-4,i))/24.D0
			!Evaluate derivs with predicted values
			call dDdv(dD2,pS)
			call dSdv(dS2,pD,rho(j,i),u(j,i))
			!Correct
			pD = D(j-1,i) + delv*(9.0D0*dD2 + 19.0D0*S(j-1,i) - 5.0D0*S(j-2,i) + S(j-3,i))/24.D0
			pS = S(j-1,i) + delv*(9.0D0*dS2 + 19.0D0*dSdvp(j-1,i) - 5.0D0*dSdvp(j-2,i) + dSdvp(j-3,i))/24.D0		
			!Evaluate derivs with predicted values
			call dDdv(dD2,pS)
			call dSdv(dS2,pD,rho(j,i),u(j,i))
			!Correct
			D(j,i) = D(j-1,i) + delv*(9.0D0*dD2 + 19.0D0*S(j-1,i) - 5.0D0*S(j-2,i) + S(j-3,i))/24.D0
			S(j,i) = S(j-1,i) + delv*(9.0D0*dS2 + 19.0D0*dSdvp(j-1,i) - 5.0D0*dSdvp(j-2,i) + dSdvp(j-3,i))/24.D0
                        if (j == jmax) then
                            dSdvp(j,i) = dS2
                        endif
		endif
	end do
	!Now solve for Q,A,Z
	do j=1,jmax
		if (j == 2) then
			!Get derivatives on previous step
			call dQdv(dQ1,D(j-1,i),S(j-1,i),Q(j-1,i),A(j-1,i),Z(j-1,i),rho(j-1,i),u(j-1,i),Lam,j-1)
                        dQdvp(j-1,i) = dQ1
			call dAdv(dA1,Z(j-1,i))
			call dZdv2(dZ1,D(j-1,i),S(j-1,i),dQ1,A(j-1,i),Z(j-1,i),rho(j-1,i),u(j-1,i),Lam,j-1)  !dZdv(dZ1,D(j-1,i),S(j-1,i),Q(j-1,i),A(j-1,i),rho(j-1,i),u(j-1,i),Lam,j-1)	
                        dZdvp(j-1,i) = dZ1
			!Predict values on next step
			pQ = Q(j-1,i) + delv*dQ1
			pA = A(j-1,i) + delv*dA1
			pZ = Z(j-1,i) + delv*dZ1
			!Evaluate derivatives with prdecited values (use values of D and S computed in previous loop)
			call dQdv(dQ2,D(j,i),S(j,i),pQ,pA,pZ,rho(j,i),u(j,i),Lam,j)
			call dAdv(dA2,pZ)
			call dZdv2(dZ2,D(j,i),S(j,i),dQ2,pA,pZ,rho(j,i),u(j,i),Lam,j) !dZdv(dZ2,D(j,i),S(j,i),pQ,pA,rho(j,i),u(j,i),Lam,j)
                        !Improve the predictor for Q (this is required because of the expected order of Q = O(v**3))
			pQ = Q(j-1,i) + delv*(3.D0*dQ1 - dQ2)/2.D0
			!Evaluate derivatives with improved prdecited values
			call dQdv(dQ2,D(j,i),S(j,i),pQ,pA,pZ,rho(j,i),u(j,i),Lam,j)
			call dZdv2(dZ2,D(j,i),S(j,i),dQ2,pA,pZ,rho(j,i),u(j,i),Lam,j) !dZdv(dZ2,D(j,i),S(j,i),pQ,pA,rho(j,i),u(j,i),Lam,j)
                        errQ = 1.0D0
                        errA = 1.0D0
                        errZ = 1.0D0
                        k = 1
			do while (abs(errQ) > tol .and. abs(errA) > tol .and. abs(errZ) > tol .and. k < maxit)
                                !Save current values
                                pQs = pQ
                                pAs = pA
                                pZs = pZ
                                !Correct
				pQ = Q(j-1,i) + 0.5D0*delv*(dQ1 + dQ2)
				pA = A(j-1,i) + 0.5D0*delv*(dA1 + dA2)
				pZ = Z(j-1,i) + 0.5D0*delv*(dZ1 + dZ2)			
				call dQdv(dQ2,D(j,i),S(j,i),pQ,pA,pZ,rho(j,i),u(j,i),Lam,j)
				call dAdv(dA2,pZ)
				call dZdv2(dZ2,D(j,i),S(j,i),dQ2,pA,pZ,rho(j,i),u(j,i),Lam,j)
                                !Compute err
                                errQ = pQs - pQ
                                errA = pAs - pA
                                errZ = pZs - pZ	
                                k = k+1
			end do
                        if (k >= maxit) then
                            write(*,*) "Exceeded max iterations"
                        endif
			!Correct with trapezoidal rule
			Q(j,i) = Q(j-1,i) + 0.5D0*delv*(dQ1 + dQ2)
			A(j,i) = A(j-1,i) + 0.5D0*delv*(dA1 + dA2)
			Z(j,i) = Z(j-1,i) + 0.5D0*delv*(dZ1 + dZ2)
		elseif (j>2 .and. j <= 4) then
			!Get derivatives on previous step
			call dQdv(dQ1,D(j-1,i),S(j-1,i),Q(j-1,i),A(j-1,i),Z(j-1,i),rho(j-1,i),u(j-1,i),Lam,j-1)
                        dQdvp(j-1,i) = dQ1
			call dAdv(dA1,Z(j-1,i))
			call dZdv2(dZ1,D(j-1,i),S(j-1,i),dQ1,A(j-1,i),Z(j-1,i),rho(j-1,i),u(j-1,i),Lam,j-1)
                        dZdvp(j-1,i) = dZ1	
			!Predict values on next step
			pQ = Q(j-1,i) + delv*(3.D0*dQ1 - dQdvp(j-2,i))/2.D0
			pA = A(j-1,i) + delv*(3.D0*dA1 - Z(j-2,i))/2.D0
			pZ = Z(j-1,i) + delv*(3.D0*dZ1 - dZdvp(j-2,i))/2.D0
			!Evaluate derivatives with prdecited values (use values of D and S computed in previous loop)
			call dQdv(dQ2,D(j,i),S(j,i),pQ,pA,pZ,rho(j,i),u(j,i),Lam,j)
			call dAdv(dA2,pZ)
			call dZdv2(dZ2,D(j,i),S(j,i),dQ2,pA,pZ,rho(j,i),u(j,i),Lam,j)
			pQ = Q(j-1,i) + 0.5D0*delv*(dQ1 + dQ2)
			pA = A(j-1,i) + 0.5D0*delv*(dA1 + dA2)
			pZ = Z(j-1,i) + 0.5D0*delv*(dZ1 + dZ2)			
			call dQdv(dQ2,D(j,i),S(j,i),pQ,pA,pZ,rho(j,i),u(j,i),Lam,j)
			call dAdv(dA2,pZ)
			call dZdv2(dZ2,D(j,i),S(j,i),dQ2,pA,pZ,rho(j,i),u(j,i),Lam,j)			
			!Correct with trapezoidal rule
			Q(j,i) = Q(j-1,i) + 0.5D0*delv*(dQ1 + dQ2)
			A(j,i) = A(j-1,i) + 0.5D0*delv*(dA1 + dA2)
			Z(j,i) = Z(j-1,i) + 0.5D0*delv*(dZ1 + dZ2)
                elseif (j > 4) then
			!Get derivatives on previous step
			call dQdv(dQ1,D(j-1,i),S(j-1,i),Q(j-1,i),A(j-1,i),Z(j-1,i),rho(j-1,i),u(j-1,i),Lam,j-1)
                        dQdvp(j-1,i) = dQ1
			call dAdv(dA1,Z(j-1,i))
			call dZdv2(dZ1,D(j-1,i),S(j-1,i),dQ1,A(j-1,i),Z(j-1,i),rho(j-1,i),u(j-1,i),Lam,j-1)
                        dZdvp(j-1,i) = dZ1	
			!Predict values on next step
			pQ = Q(j-1,i) + delv*(55.0D0*dQ1 - 59.0D0*dQdvp(j-2,i) + 37.0D0*dQdvp(j-3,i) - 9.0D0*dQdvp(j-4,i))/24.D0
			pA = A(j-1,i) + delv*(55.0D0*dA1 - 59.0D0*Z(j-2,i) + 37.0D0*Z(j-3,i) - 9.0D0*Z(j-4,i))/24.D0
			pZ = Z(j-1,i) + delv*(55.0D0*dZ1 - 59.0D0*dZdvp(j-2,i) + 37.0D0*dZdvp(j-3,i) - 9.0D0*dZdvp(j-4,i))/24.D0
			!Evaluate derivatives with prdecited values (use values of D and S computed in previous loop)
			call dQdv(dQ2,D(j,i),S(j,i),pQ,pA,pZ,rho(j,i),u(j,i),Lam,j)
			call dAdv(dA2,pZ)
			call dZdv2(dZ2,D(j,i),S(j,i),dQ2,pA,pZ,rho(j,i),u(j,i),Lam,j)
			pQ = Q(j-1,i) + delv*(9.0D0*dQ2 + 19.0D0*dQdvp(j-1,i) - 5.0D0*dQdvp(j-2,i) + dQdvp(j-3,i))/24.D0
			pA = A(j-1,i) + delv*(9.0D0*dA2 + 19.0D0*Z(j-1,i) - 5.0D0*Z(j-2,i) + Z(j-3,i))/24.D0
			pZ = Z(j-1,i) + delv*(9.0D0*dZ2 + 19.0D0*dZdvp(j-1,i) - 5.0D0*dZdvp(j-2,i) + dZdvp(j-3,i))/24.D0		
			call dQdv(dQ2,D(j,i),S(j,i),pQ,pA,pZ,rho(j,i),u(j,i),Lam,j)
			call dAdv(dA2,pZ)
			call dZdv2(dZ2,D(j,i),S(j,i),dQ2,pA,pZ,rho(j,i),u(j,i),Lam,j)			
			!Correct with trapezoidal rule
			Q(j,i) = Q(j-1,i) + delv*(9.0D0*dQ2 + 19.0D0*dQdvp(j-1,i) - 5.0D0*dQdvp(j-2,i) + dQdvp(j-3,i))/24.D0
			A(j,i) = A(j-1,i) + delv*(9.0D0*dA2 + 19.0D0*Z(j-1,i) - 5.0D0*Z(j-2,i) + Z(j-3,i))/24.D0
			Z(j,i) = Z(j-1,i) + delv*(9.0D0*dZ2 + 19.0D0*dZdvp(j-1,i) - 5.0D0*dZdvp(j-2,i) + dZdvp(j-3,i))/24.D0   
                        if (j == jmax) then
                            dQdvp(j,i) = dQ2
                            dZdvp(j,i) = dZ2
                        endif                	
		endif
                !Evaluate time derivatives of fluid variables
		call dudw(dotu,A(j,i),Z(j,i),u(j,i),up(j,i))
		ud(j,i) = dotu
		call drhodw(dotrho,D(j,i),S(j,i),Q(j,i),A(j,i),rho(j,i),rhop(j,i),u(j,i),up(j,i),j)
		!write(*,*) dotrho
		rhod(j,i) = dotrho

	end do
        !Finally we solve for drdv
end subroutine

subroutine correct(rho,rhod,u,ud,delw,D,S,Q,A,Z,rhop,up,NI,NJ,jmax,i)
	implicit none
	!Subroutine parameters
	integer, intent(in) :: NI,NJ,jmax,i
	real*8, intent(in) :: delw
	!real*8, dimension(NI), intent(in) :: rhoow
	real*8, dimension(NJ), intent(in) :: D,S,Q,A,Z,rhop,up
	real*8, dimension(NJ,NI), intent(in) :: rhod, ud
	real*8, dimension(NJ,NI), intent(inout) :: rho,u
	!Local parameters
	integer :: j
	real*8 :: dotu, dotrho, pD
	
	do j=1,jmax
		if (j==1) then
			u(j,i) = 1.D0
                        call drhodw(dotrho,D(j),S(j),Q(j),A(j),rho(j,i),rhop(j),u(j,i),up(j),j)
                        rho(j,i)  = rho(j,i-1) - 0.5D0*delw*(rhod(j,i-1) + dotrho)
		else
			call dudw(dotu,A(j),Z(j),u(j,i),up(j))
			u(j,i) = u(j,i-1) - 0.5D0*delw*(ud(j,i-1) + dotu)
                        call drhodw(dotrho,D(j),S(j),Q(j),A(j),rho(j,i),rhop(j),u(j,i),up(j),j)
                        rho(j,i)  = rho(j,i-1) - 0.5D0*delw*(rhod(j,i-1) + dotrho)
		endif
	end do
	
end subroutine

subroutine getvmaxi(v,A,vmax,vmaxi,delv,delw,NI,NJ,i,err,Flag)
	implicit none
	!Subroutine parameters
	integer, intent(in) :: NI,NJ,i
        integer, intent(out) :: Flag
	real*8, intent(in) :: delv,delw,err
	real*8, dimension(NJ), intent(in) :: v, A
	real*8, dimension(NI), intent(inout) :: vmax
	integer, dimension(NI), intent(inout) :: vmaxi
	!Loacl parameters
	integer :: j, counter, maxit, jmax
	real*8 :: vp, vprev, tol, Ap

	maxit = 10000
	tol = err
        Flag = 0
	
	!Initial guess
	vp = vmax(i-1) - 0.5*A(vmaxi(i-1))*delw
	
	vprev = 0.D0
	
	counter = 0
	
	do while(abs(vp - vprev) > tol .and. counter < maxit)
		vprev = vp
		jmax = floor(vp/delv + 1.D0)
		if (jmax < 2) then
			jmax = 2
                        Flag = 1
		elseif (jmax > NJ) then
			jmax = NJ
			Flag = 1
		endif
		Ap = A(jmax-1) + (vp-v(jmax-1))*(A(jmax) - A(jmax-1))/delv
		vp = vmax(i-1) - 0.5D0*Ap*delw
		counter = counter + 1
	end do
	if (counter >= maxit) then
                Flag = 1
	endif
	vmax(i) = vp
	vmaxi(i) = jmax

end subroutine

subroutine calcdrdv(drdv,drdvp,drdvd,D,S,A,Z,u,up,delv,delw,NI,NJ,i,jmax)
	implicit none
	!Subroutine params
	integer, intent(in) :: NI,NJ,i,jmax
	real*8, intent(in) :: delv,delw
	real*8, dimension(NJ), intent(in) :: D,S,A,Z,u,up !Pass these on current time step 
	real*8, dimension(NJ,NI), intent(inout) :: drdv, drdvp, drdvd !These must have been evaluated on the previous time step
	!Local params
	integer :: k
	!real*8, dimension(NJ) :: drdvd1, drdvd2
	!real*8, dimension(NJ) ::, drdvp

	if (i == 2) then
		!Predict drdv on next time step
		drdv(1:jmax,i) = drdv(1:jmax,i-1) - delw*drdvd(1:jmax,i-1)
		!Approximate spatial deriv with predicted value
		call d5f(drdvp,drdv,delv,i,NI,NJ,jmax)
		!Evaluate time deriv with predicted value
		call dddw(drdvd,drdv(:,i),drdvp(:,i),A,Z,u,up,NI,NJ,i,jmax)
		!do k =1,10
			!Correct drdv with trapz rule
			drdv(1:jmax,i) = drdv(1:jmax,i-1) - 0.5D0*delw*(drdvd(1:jmax,i-1) + drdvd(1:jmax,i))
			!Evaluate spatial deriv with corrected value
			call d5f(drdvp,drdv,delv,i,NI,NJ,jmax)
			!Evaluate time deriv with corrected value
			call dddw(drdvd,drdv(:,i),drdvp(:,i),A,Z,u,up,NI,NJ,i,jmax)	
		!end do
		!Correct drdv with trapz rule
		drdv(1:jmax,i) = drdv(1:jmax,i-1) - 0.5D0*delw*(drdvd(1:jmax,i-1) + drdvd(1:jmax,i))
		!Evaluate spatial deriv with corrected value
		call d5f(drdvp,drdv,delv,i,NI,NJ,jmax)
		!Evaluate time deriv with corrected value
		call dddw(drdvd,drdv(:,i),drdvp(:,i),A,Z,u,up,NI,NJ,i,jmax)	
	else
		!Predict drdv on next time step
		drdv(1:jmax,i) = drdv(1:jmax,i-1) - delw*(3.D0*drdvd(1:jmax,i-1) - drdvd(1:jmax,i-2))/2.D0
		!Approximate spatial deriv with predicted value
		call d5f(drdvp,drdv,delv,i,NI,NJ,jmax)
		!Evaluate time deriv with predicted value
		call dddw(drdvd,drdv(:,i),drdvp(:,i),A,Z,u,up,NI,NJ,i,jmax)
		!do k =1,10
			!Correct drdv with trapz rule
			drdv(1:jmax,i) = drdv(1:jmax,i-1) - 0.5D0*delw*(drdvd(1:jmax,i-1) + drdvd(1:jmax,i))
			!Evaluate spatial deriv with corrected value
			call d5f(drdvp,drdv,delv,i,NI,NJ,jmax)
			!Evaluate time deriv with corrected value
			call dddw(drdvd,drdv(:,i),drdvp(:,i),A,Z,u,up,NI,NJ,i,jmax)	
		!end do
		!Correct drdv with trapz rule
		drdv(1:jmax,i) = drdv(1:jmax,i-1) - 0.5D0*delw*(drdvd(1:jmax,i-1) + drdvd(1:jmax,i))
		!Evaluate spatial deriv with corrected value
		call d5f(drdvp,drdv,delv,i,NI,NJ,jmax)
		!Evaluate time deriv with corrected value
		call dddw(drdvd,drdv(:,i),drdvp(:,i),A,Z,u,up,NI,NJ,i,jmax)		
	endif
end subroutine

subroutine geotr(t,r,u,up,X,Xr,delv,wi,NI,NJ,i,jmax)
	implicit none
	!Subroutine params
	integer, intent(in) :: NI,NJ,i, jmax
	real*8, intent(in) :: delv,wi
	real*8, dimension(NJ), intent(in) :: u,up,X,Xr
	real*8, dimension(NJ,NI), intent(inout) :: t,r
	!Local params
	integer :: j
	real*8 :: tp1,tpp1,rp1,rpp1  !Derivs on previous step
	real*8 :: tp2,tpp2,rp2,rpp2  !Predicted derivs
	real*8 :: pt,pdt,pr,pdr  !Predicted values
	real*8, dimension(NJ,NI) :: dt,dr !These are local for now, we dont use them anywhere

	dt = 0.D0
	dr = 0.D0
	
	!Set IC's	
	t(1,i) = wi
	dt(1,i) = -1.D0
	r(1,i) = 0.D0
	dr(1,i) = 1.D0/X(1)
	
	do j=2,jmax
		!Get derivatives on previous step
		call dy1(tp1,dt(j-1,i))
		call dy2(tpp1,dt(j-1,i),u(j),up(j))
		call dy3(rp1,dr(j-1,i))
		call dy4(rpp1,dt(j-1,i),dr(j-1,i),u(j),up(j),X(j),Xr(j))
		
		!Predict
		pt = t(j-1,i) + delv*tp1
		pdt = dt(j-1,i) + delv*tpp1
		pr = r(j-1,i) + delv*rp1
		pdr = dr(j-1,i) + delv*rpp1
		
		!Evaluate with predicted values
		call dy1(tp2,pdt)
		call dy2(tpp2,pdt,u(j),up(j))
		call dy3(rp2,pdr)
		call dy4(rpp2,pdt,pdr,u(j),up(j),X(j),Xr(j))
		
		!Correct
		t(j,i) = t(j-1,i) + 0.5D0*delv*(tp1 + tp2)
		dt(j,i) = dt(j-1,i) + 0.5D0*delv*(tpp1 + tpp2)
		r(j,i) = r(j-1,i) + 0.5D0*delv*(rp1 + rp2)
		dr(j,i) = dr(j-1,i) + 0.5D0*delv*(rpp1 + rpp2)		
	end do
end subroutine
