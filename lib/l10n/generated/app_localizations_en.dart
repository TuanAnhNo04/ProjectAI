import 'app_localizations.dart';

/// The translations for English (`en`).
class AppLocalizationsEn extends AppLocalizations {
  AppLocalizationsEn([String locale = 'en']) : super(locale);

  @override
  String get appTitle => 'MedAssist';

  @override
  String get loginTitle => 'Sign in';

  @override
  String get loginSubtitle => 'Medicine information you can rely on';

  @override
  String get usernameLabel => 'Username';

  @override
  String get passwordLabel => 'Password';

  @override
  String get emailLabel => 'Email';

  @override
  String get signIn => 'Sign in';

  @override
  String get signUp => 'Sign up';

  @override
  String get signUpTitle => 'Create an account';

  @override
  String get signUpLink => 'Create an account';

  @override
  String get showPassword => 'Show password';

  @override
  String get hidePassword => 'Hide password';

  @override
  String get signInFailed => 'Sign-in failed.';

  @override
  String get signUpFailed => 'Sign-up failed.';

  @override
  String get processFailed => 'Could not analyse the image.';

  @override
  String get createAccount => 'Create an account';

  @override
  String get signOut => 'Sign out';

  @override
  String get signOutConfirmTitle => 'Sign out?';

  @override
  String get signOutConfirmBody =>
      'You will need to sign in again to continue.';

  @override
  String get cancel => 'Cancel';

  @override
  String get confirm => 'Confirm';

  @override
  String get delete => 'Delete';

  @override
  String get processImageTitle => 'Scan medicine';

  @override
  String get pickImage => 'Pick image';

  @override
  String get processImage => 'Analyse';

  @override
  String get noImagePicked => 'No image selected yet';

  @override
  String get history => 'History';

  @override
  String get search => 'Search';

  @override
  String get resultsTitle => 'Results:';

  @override
  String get noResults => 'No results.';

  @override
  String get processSuccess => 'Done!';

  @override
  String get searchTitle => 'Search medicine';

  @override
  String get searchHint => 'Medicine name';

  @override
  String get searchEmptyQuery => 'Please enter a medicine name.';

  @override
  String get historyTitle => 'Processing history';

  @override
  String get historyEmpty => 'No processing history';

  @override
  String get historyLoadFailed => 'Could not load history.';

  @override
  String get searchResultsLabel => 'Search results:';

  @override
  String get historyFile => 'File:';

  @override
  String get deleteRecordTitle => 'Delete this record?';

  @override
  String get deleteRecordBody =>
      'This record will be removed from your history.';

  @override
  String get deleted => 'Deleted';

  @override
  String get deleteFailed => 'Could not delete the record';

  @override
  String get retry => 'Retry';

  @override
  String get offline => 'Offline';

  @override
  String offlineUpdated(String time) {
    return 'Offline · updated $time';
  }

  @override
  String get noInformation => 'No information';

  @override
  String get medicineName => 'Medicine name';

  @override
  String get composition => 'Ingredients';

  @override
  String get uses => 'Uses';

  @override
  String get sideEffects => 'Side effects';

  @override
  String get manufacturer => 'Manufacturer';

  @override
  String get reviewExcellent => 'Excellent %';

  @override
  String get reviewAverage => 'Average %';

  @override
  String get reviewPoor => 'Poor %';

  @override
  String get validationUsernameRequired => 'Please enter your username.';

  @override
  String get validationPasswordRequired => 'Please enter your password.';

  @override
  String validationPasswordTooShort(int length) {
    return 'Password must be at least $length characters.';
  }

  @override
  String get validationEmailRequired => 'Please enter your email address.';

  @override
  String get validationEmailInvalid => 'That email address is not valid.';

  @override
  String get medicalDisclaimer =>
      'For reference only. This does not replace advice from a doctor or pharmacist.';
}
