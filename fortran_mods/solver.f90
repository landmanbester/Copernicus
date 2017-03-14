!=================================================================	  
! PROGRAM TO SOLVE THE CHARACTERISTIC INITIAL VALUE PROBLEM
!==================================================================
include 'model.f90'
include 'support.f90'
include 'stencils.f90'
!This is just for compatibility between windows and linux
program civp
	implicit none
	integer :: NI, NJ
	character(8) :: date
	character(10) :: time
	call date_and_time(DATE=date, TIME=time)
	!DEC$ IF DEFINED(_WIN32)
	call system('cls')
	!DEC$ ELSEIF DEFINED(__linux)
	call system('clear')
	!DEC$ ENDIF
end program
!---------------------------------------------------------------
! Solve CIVP
!---------------------------------------------------------------
subroutine solve(v,delv,w,delw,ui,rhoi,Lam & !These are all the inputs
	,D,S,Q,A,Z,rho,rhod,rhop,u,ud,up,upp,vmax,vmaxi,r,t,X,dXdr,drdv,drdvp,dSdvp&
        ,dQdvp,dZdvp,LLTBCon,Dww,Aw,T1,T2,sigmasq,err,NI,NJ,Flag2) !These are all outputs except NI and NJ which are optional
!f2py threadsafe
	implicit none
	!Subroutine parameters
	integer, intent(in) :: NI, NJ
	real*8, intent(in) :: Lam,delv,delw,err
        real*8, dimension(NI,NJ), intent(in)  :: w
	real*8, dimension(NJ), intent(in) :: v,ui,rhoi
	real*8, dimension(NI), intent(out) :: vmax
	integer, dimension(NI), intent(out) :: vmaxi
        integer, intent(out) :: Flag2
	real*8, dimension(NJ,NI), intent(out) :: D,S,Q,A,Z,rho,rhod,rhop,u,ud,up,upp,X,dXdr,T1,T2,sigmasq
        real*8, dimension(NJ,NI), intent(out) :: t,r,drdv,drdvp,dSdvp,dQdvp,dZdvp,LLTBCon,Dww,Aw
	!Local variables
	integer :: i, jmax, k, Flag
	real*8 :: dfdr5 !For the derivative function
	real*8, dimension(NJ,NI) :: dtdw, dwdt, dtdv, dvdt !For partial derivs involving t 
	real*8, dimension(NJ,NI) :: drdw, dwdr, dvdr, drdvd !For partial derivs involving r 
	!real*8, dimension(NJ,NI) :: dt, dr

	!init output arrays and set boundary conditions
        Flag = 0
        Flag2 = 0
	vmaxi = 0
	vmaxi(1) = NJ
	vmax = 0.D0
	vmax(1) = v(NJ)
	D = 0.D0
	S = 0.D0
        dSdvp = 0.D0
	Q = 0.D0
        dQdvp = 0.D0
	A = 0.D0
	Z = 0.D0
        dZdvp = 0.D0
	rho = 0.D0
	rho(:,1) = rhoi
	rhod = 0.D0
	rhop = 0.D0
	u = 0.D0
	u(:,1) = ui
	ud = 0.D0
	up = 0.D0	
	upp = 0.D0
	t = 0.D0
	r = 0.D0
	X = 0.D0
	dXdr = 0.D0
        drdvp = 0.D0
	jmax = NJ
        LLTBCon = 0.D0
        Dww = 0.0D0
        Aw = 0.0D0
        T1 = 0.0D0
        T2 = 0.0D0
        sigmasq = 0.0D0 

        !To store partial derivs involving t
	dtdw = 0.D0
	dwdt = 0.D0
	dtdv = 0.D0
	dvdt = 0.D0
	
	!To store partial derivs involving r
	drdv = 0.D0
	drdw = 0.D0
        dvdr = 0.D0
        dwdr = 0.D0

        !Get soln on PLC0
        call evaluate(rho,u,rhod,ud,Lam,delv,D,S,Q,A,Z,rhop,up,upp,NI,NJ,jmax,1,dSdvp,dQdvp,dZdvp)

        !Do Tests of CP
        call shear_test(T1,u(:,1),up(:,1),D(:,1),S(:,1),Q(:,1),A(:,1),vmaxi,1,NI,NJ)
        call curve_test(T2,D(:,1),S(:,1),dSdvp(:,1),u(:,1),up(:,1),upp(:,1),vmaxi,1,NI,NJ)
        call get_sigmasq(sigmasq,u(:,1),up(:,1),ud(:,1),D(:,1),S(:,1),Q(:,1),A(:,1),Z(:,1),vmaxi,1,NI,NJ)

	!Get partial_t components
	call transt(dtdw,dwdt,dtdv,dvdt,A(:,1),u(:,1),NI,NJ,1,jmax)

        !Set initial values of drdv, drdvp and drdvd (by fixing the gauge r = uD)
        drdv(:,1) = up(:,1)*D(:,1) + u(:,1)*S(:,1)
        drdvp(:,1) = upp(:,1)*D(:,1) + 2.0D0*up(:,1)*S(:,1) + u(:,1)*dSdvp(:,1)
        call dddw(drdvd,drdv(:,1),drdvp(:,1),A(:,1),Z(:,1),u(:,1),up(:,1),NI,NJ,1,jmax)
        
	!Get remaining partial_r components
	call transr(drdw,dwdr,drdv,dvdr,A(:,1),u(:,1),NI,NJ,1,jmax)

	!Get X and dXdr
	call getXandXr(X,dXdr,dwdr(:,1),dvdr(:,1),drdv(:,1),drdvp(:,1),drdvd(:,1),u(:,1),up(:,1),ud(:,1),NI,NJ,1,jmax)

        if (NI > 1) then

           !Solve for remaining domain
           do i=2,NI
              !Predict fluid variables on next PLC
              call predict(delw,D(:,i-1),S(:,i-1),Q(:,i-1),A(:,i-1),Z(:,i-1),rhop(:,i-1),up(:,i-1),NI,NJ,jmax,i,rho,u,rhod,ud)
              !Evaluate hypersurface variables with the predicted values
              call evaluate(rho,u,rhod,ud,Lam,delv,D,S,Q,A,Z,rhop,up,upp,NI,NJ,jmax,i,dSdvp,dQdvp,dZdvp)

              !Get the predicted characteristic cut off
              call getvmaxi(v,A(:,i),vmax,vmaxi,delv,delw,NI,NJ,i,err,Flag)
              if (Flag==1) then
                 Flag2 = 1
                 exit
              endif
              jmax = vmaxi(i)

              if (jmax > NJ) then
                 Flag2 = 1
                 write(*,*) "Warning! Got jmax > NJ in first estimate in main. Rejecting this sample"
                 exit
              endif

              !Do iterative step (5 was chosen at random, most of the time there is not much improvement beyond 5 iterations)
              do k=1,5 
                 !Correct fluid variables
                 call correct(rho,rhod,u,ud,delw,D(:,i),S(:,i),Q(:,i),A(:,i),Z(:,i),rhop(:,i),up(:,i),NI,NJ,jmax,i)

                 !Evaluate hypersurface variables with the corrected values
                 call evaluate(rho,u,rhod,ud,Lam,delv,D,S,Q,A,Z,rhop,up,upp,NI,NJ,jmax,i,dSdvp,dQdvp,dZdvp)

              end do

!!$              !Re-evaluate vmaxi with corrected value
!!$              call getvmaxi(v,A(:,i),vmax,vmaxi,delv,delw,NI,NJ,i,err,Flag)
!!$              if (Flag==1) then
!!$                 Flag2 = 1
!!$                 exit
!!$              endif
!!$              jmax = vmaxi(i)
!!$
!!$              if (jmax > NJ) then
!!$                 Flag2 = 1
!!$                 write(*,*) "Warning! Got jmax > NJ in corrrected estimate in main. Rejecting this sample"
!!$                 exit
!!$              endif

              !Final correct
              call correct(rho,rhod,u,ud,delw,D(:,i),S(:,i),Q(:,i),A(:,i),Z(:,i),rhop(:,i),up(:,i),NI,NJ,jmax,i)

              !Do Tests of CP
              call shear_test(T1,u(:,i),up(:,i),D(:,i),S(:,i),Q(:,i),A(:,i),vmaxi,i,NI,NJ)
              call curve_test(T2,D(:,i),S(:,i),dSdvp(:,i),u(:,i),up(:,i),upp(:,i),vmaxi,i,NI,NJ)
              call get_sigmasq(sigmasq,u(:,i),up(:,i),ud(:,i),D(:,i),S(:,i),Q(:,i),A(:,i),Z(:,i),vmaxi,i,NI,NJ)

              !Get partial derivs involving t
              call transt(dtdw,dwdt,dtdv,dvdt,A(:,i),u(:,i),NI,NJ,i,jmax)

              !Solve for  drdv
              call calcdrdv(drdv,drdvp,drdvd,D(:,i),S(:,i),A(:,i),Z(:,i),u(:,i),up(:,i),delv,delw,NI,NJ,i,jmax)

              !Get the remaining coord transform
              call transr(drdw,dwdr,drdv,dvdr,A(:,i),u(:,i),NI,NJ,i,jmax)

              !Get X and dXdr
              call getXandXr(X,dXdr,dwdr(:,i),dvdr(:,i),drdv(:,i),drdvp(:,i),drdvd(:,i),u(:,i),up(:,i),ud(:,i),NI,NJ,i,jmax)
!!$		write(*,*) "Got this far", i
!!$		!Solve geodesic equations
!!$		call geotr(t,r,u(:,i),up(:,i),X(:,i),dXdr(:,i),delv,w(i),NI,NJ,i,jmax)

           end do
        endif
!!$        if (Flag2 /= 1) then
!!$            !Get the LLTB consistency relation
!!$            call get_LLTBCon(LLTBCon,Dww,Aw,D,S,Q,A,Z,dZdvp,u,rho,Lam,delw,NI,NJ,vmaxi)
!!$	endif
end subroutine
