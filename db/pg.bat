@echo off
rem PostgreSQL 17.6 portable: start / stop helper.
rem This is NOT a Windows service, so it does not start on boot.
rem usage:  db\pg.bat start   |   db\pg.bat stop   |   db\pg.bat status
rem Keep this file pure ASCII. cmd.exe reads .bat in the OEM codepage (936 on
rem Chinese Windows); UTF-8 Chinese comments can swallow the line break and
rem turn every line into one giant broken command.
setlocal
set PG=F:\pgsql\pgsql\bin
set DATA=F:\pgsql\data

if "%~1"=="start"  goto start
if "%~1"=="stop"   goto stop
if "%~1"=="status" goto status
echo usage: db\pg.bat start^|stop^|status
exit /b 1

:start
"%PG%\pg_ctl.exe" -D "%DATA%" -l "F:\pgsql\server.log" start
goto :eof

:stop
"%PG%\pg_ctl.exe" -D "%DATA%" stop
goto :eof

:status
"%PG%\pg_isready.exe" -h 127.0.0.1 -p 5432
goto :eof
