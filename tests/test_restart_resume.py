import subprocess
import sys


def _run_help(args: list[str]) -> str:
	p = subprocess.run([sys.executable, "-m", "src.main", *args], capture_output=True, text=True)
	# Certains environnements écrivent l'aide sur stdout
	out = (p.stdout or "") + (p.stderr or "")
	return out.lower()


def test_cli_help_lists_core_commands():
	out = _run_help(["--help"])
	# Commandes anglaises
	assert "start" in out and "stop" in out and "status" in out and "settings" in out and "predict" in out
	# Alias français
	assert "demarrer" in out and "arreter" in out and "etat" in out and "parametres" in out and "prevoir" in out


def test_cli_prevoir_help_has_french_options():
	out = _run_help(["prevoir", "--help"])
	assert "--unite" in out and "--valeur" in out and "--montant-eur" in out
