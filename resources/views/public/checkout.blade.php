@extends('layouts.custFeane')

@section('title', 'Quickbite - Checkout')

@section('cart')
<section class="cart_section layout_padding" style="margin-top: 100px;">
    <div class="container">
        <div class="heading_container heading_center">
            <h2>Konfirmasi Checkout</h2>
        </div>

        @if(isset($cart) && count($cart) > 0)
            <div class="table-responsive mt-4">
                <table class="table table-bordered">
                    <thead>
                        <tr class="table-light">
                            <th>Produk</th>
                            <th>Ukuran</th>
                            <th>Jumlah</th>
                            <th>Harga Satuan</th>
                            <th>Subtotal</th>
                        </tr>
                    </thead>
                    <tbody>
                        @foreach($cart as $item)
                            <tr>
                                <td>{{ $item['name'] }}</td>
                                <td>{{ ucfirst($item['portion_size'] ?? 'normal') }}</td>
                                <td>{{ $item['quantity'] }}</td>
                                <td>Rp {{ number_format($item['price'], 0, ',', '.') }}</td>
                                <td>Rp {{ number_format($item['price'] * $item['quantity'], 0, ',', '.') }}</td>
                            </tr>
                        @endforeach
                    </tbody>
                </table>
            </div>

            <form action="{{ route('checkout.process') }}" method="POST" class="mt-4">
                @csrf

                <div class="row">
                    <div class="col-md-6">
                        <label for="order_type" class="form-label">Tipe Pesanan</label>
                        <select name="order_type" id="order_type" class="form-select" required>
                            <option value="dine_in">Makan di Tempat</option>
                            <option value="take_away">Dibungkus</option>
                        </select>
                    </div>
                    <div class="col-md-6">
                        <label for="payment_type" class="form-label">Metode Pembayaran</label>
                        <select name="payment_type" id="payment_type" class="form-select" required>
                            <option value="QRIS">QRIS</option>
                            <option value="debit_card">Kartu Debit</option>
                            <option value="credit_card">Kartu Kredit</option>
                            <option value="e_wallet">E-Wallet</option>
                        </select>
                    </div>
                </div>

                <div class="text-end mt-4">
                    <h4>Total Belanja: <span class="text-primary">Rp {{ number_format($total, 0, ',', '.') }}</span></h4>
                    <button type="submit" class="btn btn-success mt-3">Proses Checkout <i class="fa fa-arrow-right"></i></button>
                </div>
            </form>
        @else
            <div class="alert alert-info text-center mt-4">
                <h4>Keranjang belanja Anda masih kosong.</h4>
                <a href="{{ route('public.menus.index') }}" class="btn btn-primary mt-3">Lihat Menu Kami</a>
            </div>
        @endif
    </div>
</section>
@endsection