/**
 * 1. Turiste sortiramo naraščajoče po njihovi teži
 * 2. Ustvarimo kazalca low = 0, high = n-1 in števec kajakov count = 0
 * 3. While loop dokler low <= high 
 *    a) najtežji turist dobi kajak count += 1
 *    b) če teza[low] + teza[high] <= k, dodaj najlažjega turista low += 1
 *    c) vedno zmanjšaj high -= 1
 * 6. vrni count kot število kajakov
 *
 * Časovna zahtevnost:
 *   - sortiranje: O(nlogn)
 *   - loop skozi urejen seznam: O(n)
 *   - SKUPAJ: O(nlogn)
 *
 * Prostorska zahtevnost: O(n) ker lahko uredimo seznam na mestu
 *
 * Dokaz minimalnega števila kajakov:
 *   - PREDNOST: algoritem najprej dodeli najtežjega turista, saj ga ni mogoče združiti z nikomer težjim
 *   - ZAMENJAVA: če obstaja boljša rešitev, jo lahko izboljšamo tako, da vedno združimo najtežjega
 *     in najlažjega turista, kadar je to mogoče, kar pokaže, da algoritem ne naredi napak
 *   - STRUKTURA: Problem ima optimalno podstukturo - če odstranimo najtežjega (in najlažjega, če ju združimo),
 *     ostane enak problem na manjšem seznamu, ki ga algoritem prav tako reši optimalno
 */
